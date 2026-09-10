# vLLM Custom Docker Build

This directory builds the base vLLM Docker image. It does **not** run models directly — model containers are launched via `vllm.sh` in this directory (aliased as `vllm` in `~/.bash_aliases`), which uses this image as its base.

The image targets an RTX 5090 (Blackwell, compute 12.0) with cutting-edge CUDA drivers.

## Documentation

General information goes to CLAUDE.md; information related to `vllm.sh` goes right into the script file (as comments).

Documentation must be kept in sync and useful. Use terse language, avoid fluffy or chatty sections. Focus on the "why" if the "how" is obvious from looking at the code. Target audience is a coding agent such as yourself.

## Why a custom build?

We build vLLM **from current upstream source** on top of NVIDIA's `nvcr.io/nvidia/vllm` container so we can track bleeding-edge vLLM (features/fixes ahead of any NVIDIA release) while keeping NVIDIA's Blackwell-tuned CUDA/PyTorch runtime. The official `vllm/vllm-openai:latest` image lags and its CUDA/torch combo isn't matched to the RTX 5090 + our driver, so a custom build is needed to reconcile NVIDIA's base torch with the newer vLLM source and its CUDA-extension deps.

> **HISTORY:** The original rationale (below, kept for context) was about **driver 580.x**, which capped at CUDA 13.0 and forced an `nvcc 13.0` override + the `26.03` base pin. The driver is now **610.43.02 (CUDA 13.3)**, so on **2026-07-17** the base was bumped to **`26.06-py3`** (torch 2.13, CUDA 13.3) and all the driver-580 workarounds were deleted — see "Base image (26.06)" below. The paragraphs that follow describe the retired 580.x constraints.
>
> _(Retired 580.x rationale)_ The official image used CUDA 12.9, incompatible with driver 580.x; standard PyTorch cu130 triggered forward-compat error 803. Driver 580.x needed the CUDA 13.1 runtime (only in NVIDIA's internal PyTorch), vLLM's CUDA extensions needed full dev headers absent from the NVIDIA container, and nvcc 13.1 generated PTX driver 580.x couldn't run — so we substituted the CUDA 13.0 compiler.

## Hardware

| Component | Details |
|-----------|---------|
| GPU | RTX 5090 (Blackwell, sm_120) |
| Driver | 610.43.02 (CUDA 13.3 runtime; UMD 13.3) |
| CPU | Ryzen 9 9900X (12-core / 24 threads) |
| RAM | ~91 GB |

## Files

| File | Purpose |
|------|---------|
| `update.sh` | Updates `vllm/` to latest upstream (fast-forward only) and kicks off an async build. Automates the mechanical part of `/latest`; does **not** watch or fix the build |
| `build-async.sh` | **Preferred build script** — detaches from terminal, survives SSH disconnects |
| `build.sh` | Synchronous build script (called internally by `build-async.sh`) |
| `Dockerfile` | 2-stage Dockerfile, base `26.06-py3` (see below) |
| `Dockerfile.26.03` | Archived previous 3-stage build (driver-580.x era); kept for reference/rollback |
| `PLAN-base-26.06.md` | Base-bump plan + measured dependency audit |
| `vllm/` | vLLM source, tracked as a **git submodule** → `github.com/vllm-project/vllm` (pins the built commit) |
| `vllm.sh` | Interactive Docker model menu (aliased as `vllm`); launches model containers from `vllm-custom:latest` |
| `build.number` | Persisted monotonic build counter; `build.sh` increments it (on success only) and tags the image `vllm-custom:b<N>` |
| `build.history` | Append-only log of successful builds: build number, date/time, and vLLM version. Distinct from `build.log` (full build output) |
| `bench-inference.ts` | Inference benchmark against a **running** server (TTFT, decode tok/s, MTP acceptance). Node-native TS, no deps. Appends each run to `bench-results/bench.jsonl`; `--report` prints the saved history as a table |
| `bench-results/` | Gitignored benchmark history (JSONL, one record per run). Read it with `bench-inference.ts --report` |

## Dockerfile strategy (2 stages, base `26.06-py3`)

1. **`builder`** — starts from `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13, CUDA 13.3) and builds vLLM from current source into a wheel using the base's **native nvcc 13.3**.
2. **Runtime** — fresh `nvcr.io/nvidia/vllm:26.06-py3` base, installs the wheel (`--no-deps` to preserve NVIDIA's custom torch), then pins the handful of runtime deps that current source needs newer than the base's vLLM 0.22.1.

> **26.03 → 26.06 bump (2026-07-17):** the old `cuda-devel` stage + nvcc-13.0 override, the `register_opaque_type` hoist sed, and the `cuda_view.cu` patch were all **deleted** — verified unnecessary on torch 2.13 / driver 610. The previous 3-stage 26.03 Dockerfile is archived as `Dockerfile.26.03`; see `PLAN-base-26.06.md` for the measured dep audit. Validated by building b14 and serving `vllm.sh 2` (Qwen NVFP4 MoE + MTP) end-to-end.

### Pins the runtime stage still needs (source ahead of 26.06's vLLM 0.22.1)
Because we build bleeding-edge source on a base tuned for 0.22.1, these stay pinned (guards fail-fast if a symbol/torch regresses): `transformers==5.12.1`, `xgrammar==0.2.3`, `flashinfer-python/-cubin==0.6.18` (+`--extra-index-url https://flashinfer.ai/whl/`), `compressed-tensors==0.17.0`, `humming-kernels==0.1.10`, `tokenspeed-mla==0.1.8`, `tilelang==0.1.9`, `nvidia-cutlass-dsl==4.6.0` + `quack-kernels==0.6.1` (matched cute-DSL pair), `apache-tvm-ffi==0.1.10`, `nvidia-cudnn-frontend>=1.19.1`, `mistral_common>=1.11.5`. The base bump **dropped** the previously-needed `openai` and `fastsafetensors` pins (26.06 satisfies them natively).

**The flashinfer pin tracks `vllm/requirements/cuda.txt` — diff it on every update.** It is the one pin upstream moves *for* us, and a stale one is not caught by the build: b19 (2026-09-04) built clean against 0.6.14 while source had moved to 0.6.18, then crash-looped every config at cudagraph capture with `TypeError: xqa_batch_decode_with_kv_cache() got an unexpected keyword argument 'q_cu_seq_lens'`. The XQA decode path is gated on `is_device_capability_family(120)`, so it is **sm_120 / this box only** — upstream CI does not cover it, and the failure surfaces at *runtime on the first model start*, not at build time. The `RUN` block now guards `q_cu_seq_lens` alongside `kv_cache_sf`; add a guard for whatever kwarg the next skew lands on rather than trusting the version number.

## Transformers pin (load-bearing) & Gemma 4 MTP

The runtime stage installs the vLLM wheel with `--no-deps` (to preserve NVIDIA's CUDA 13.1 torch), so **transformers is never pulled as a wheel dependency** — the version is whatever the explicit `RUN pip install transformers==<X>` line sets. Removing that line leaves the base image's transformers **4.57.5**, which is 4.x and doesn't recognize the `gemma4` architecture at all → every Gemma 4 config breaks at model load. The pin is required for all Gemma 4 options, not optional.

**Pinned to `transformers==5.12.1`** (bumped from 5.5.4 on 2026-07-02) to enable Gemma 4 MTP:

- Gemma 4 MTP (Multi-Token Prediction) speculative decoding uses Google's drafter `google/gemma-4-31B-it-assistant` (~0.5B), enabled by `vllm.sh` configs **`4m`** (105k ctx) and **`4`** (the same stack at 90k ctx + a 24 GiB host-RAM KV tier; see below), with a KV budget sized so the drafter weights fit at gpu-mem-util 0.982. **`vllm.sh`'s `4m` block is the source of truth for these numbers — it carries the full dated tuning log. Do not trust figures quoted here over that block.** Current: `CONTEXT_SIZE=105000` with `--num-gpu-blocks-override 6780`.
  - **The tuning lever changed on 2026-08-03.** Before that, the KV pool was whatever vLLM's profiler handed out and the only lever was trimming `CONTEXT_SIZE` under it. Since then the pool is **pinned** by `--num-gpu-blocks-override`, deliberately claiming more than the profiler offers (the profiler under-estimates because the torch.compile working set is still resident when the peak is measured) and making warm and cold boots identical. So **re-tuning now means adjusting the block override, not just lowering context.**
  - **Both directions are load-bearing, and they fail differently.** Too few blocks → the old failure: KV cache too small at startup, engine won't boot. Too many → the pool eats the activation headroom and the engine **boots fine, serves for hours, then dies mid-request** with a CUDA OOM on a prefill buffer, which `--restart unless-stopped` silently masks as a restart. Check `RestartCount`, not just health.
  - History: ran **128k** until the 2026-07-17 vLLM update (Qwen 3.6 + new default kernels/cudagraphs) dropped the pool to ~6.86 GiB (~85k tokens) and 128k OOMed at startup → retuned to `83968`. Raised to **105000** on 2026-08-03 once the 7000-block override pinned the pool at 119,182 tokens. Override cut to **6780** on 2026-09-09 (115,437 tokens, 1.10x) after b21's larger non-KV footprint left only 154.88 MiB free and a 168 MiB prefill buffer began crashing the engine hours into serving.
  - **Re-derive after every image bump — this is routinely forgotten** (it was skipped across b18→b21, which is what caused the 2026-09-09 crashes). Boot cold, read `GPU KV cache size: <N> tokens`, and confirm real headroom via `nvidia-smi` — note the `gpu_worker.py:871` "kv cache memory in use" line reports the *profiled* figure, not the pinned pool, so it does not reflect the override. All of this is a `vllm.sh`-only fix; no image rebuild.
  - **Config `4` has its own override (6400); never sync it to `4m`'s.** Its RAM KV tier forces it to run without expandable_segments, which costs ~500 MiB of headroom. Re-derive it separately after an image bump, using a long-prompt stress test rather than idle headroom. Details are in the `vllm.sh` `4` block.
- The drafter's `config.json` has `model_type: gemma4_assistant`. vLLM does **not** register this in its config registry (true even on upstream main) — it relies on **transformers** to recognize it, and that model_type only landed in **transformers 5.12.x**. `5.7.0` does NOT have it (despite the drafter stamping `5.7.0.dev0`); with anything <5.12 the engine crashes at config load: "Transformers does not recognize `gemma4_assistant`".
- Validated 2026-07-02: 5.12.1 boots cleanly against our wheel, preserves NVIDIA's torch, and drives the drafter at ~80–91% draft acceptance (mean accept length ~1.8 → near ~1.8× decode). MTP is text-only (drafter has no vision tower); image prompts still run on the target.

**The 26.06 base does NOT fix this** — it ships **transformers 5.6.0** (below 5.12.1, no `gemma4_assistant`), so the explicit `transformers==5.12.1` pin is still required after the base bump. (The base bump itself was done on 2026-07-17 and did retire the nvcc override / `cuda_view.cu` / hoist patches — see "Dockerfile strategy" above — but the transformers pin is independent of the base and stays.)

## Building

> **Routine update + build:** run `./update.sh` — it fast-forwards `vllm/` to upstream `main` (aborts on a non-fast-forward or a dirty tree; never forces), reports the version delta, and starts a detached build. Pass `--no-build` to update only. It does **not** block on the build and does **not** diagnose failures — a large upstream jump can break a grep-guarded Dockerfile patch or pin, which needs a human/agent (the `/latest` skill covers watch-and-fix).

> **Always use `build-async.sh`**, not `build.sh` directly. It survives SSH disconnects, captures the exit code, and supports agent-friendly polling.

```bash
./build-async.sh start       # start detached build
./build-async.sh status      # human-readable status + last 20 log lines
./build-async.sh status --json  # machine-readable JSON (for agents)
./build-async.sh wait        # block until done, exits with build's exit code
./build-async.sh log         # tail -f the live log
./build-async.sh stop        # kill the running build
```

Blocking agent workflow:
```bash
./build-async.sh start && ./build-async.sh wait
```

Key output files (in this directory):

| File | Contents |
|------|----------|
| `build.log` | Full stdout+stderr of the build |
| `build.pid` | PID of running build (deleted on completion) |
| `build.exit` | Exit code on completion (`0` = success) |

Grep-able log markers: `BUILD_STARTED`, `BUILD_SUCCESS`, `BUILD_FAILED`

Build args (editable in `build.sh`):

| Arg | Default | Notes |
|-----|---------|-------|
| `TORCH_CUDA_ARCH_LIST` | `12.0` | Blackwell only — keeps build fast |
| `MAX_JOBS` | `16` | Leaves 8 threads for the system |
| `NVCC_THREADS` | `4` | Parallelism inside nvcc |

Expected build time: 60 minutes. Final image: ~15 GB.

## Output image tags

Every build applies three tags:

- `vllm-custom:b<N>` — monotonically increasing build number (from `build.number`, bumped on success only)
- `vllm-custom:<git-describe>` — vLLM's own version, e.g. `vllm-custom:v0.22.1rc0-19-g50c80d79230`
- `vllm-custom:latest` — always points at the most recent build

## After building

1. Verify: `docker images | grep vllm-custom`
2. No tag edit needed — `vllm.sh` is pinned to `vllm-custom:latest`, which always tracks the newest build. (Pin to a specific `b<N>` tag only if you need to roll back.)
3. Test: `./vllm.sh rec 4` (removes and re-creates the vLLM docker container for Gemma4:31b)

## Base image bump to `26.06-py3` (DONE 2026-07-17, b14)

The base was moved from `26.03.post1` to `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13, CUDA 13.3, NVIDIA vLLM 0.22.1) on driver 610. Outcome:

- Deleted the `cuda-devel` stage, nvcc-13.0 override, `register_opaque_type` hoist sed, and `cuda_view.cu` patch — all verified unnecessary on torch 2.13 (register_opaque_type has `hoist` natively; `cuda_view.cu` compiles unpatched against 2.13's stable ABI).
- Dropped the `openai==2.45.0` and `fastsafetensors` pins (26.06 satisfies them).
- Kept `transformers==5.12.1` and the CUDA/kernel pins listed under "Dockerfile strategy".
- Fixed the cutlass/quack `ThrMma` crash via the matched `nvidia-cutlass-dsl==4.6.0` + `quack-kernels==0.6.1` pair.

Validated: b14 serves `vllm.sh 2` (Qwen NVFP4 MoE + MTP) end-to-end. `Dockerfile.26.03` archives the previous build; `PLAN-base-26.06.md` has the full audit/rationale.

> **Reminder for the next base bump:** we track bleeding-edge vLLM source, so *any* fixed NVIDIA base will lag some deps — expect to keep a set of CUDA/kernel pins in the runtime stage no matter what. Trial in a separate `Dockerfile.<ver>`, diff shipped versions against `vllm/requirements/{common,cuda}.txt` (see `PLAN-base-26.06.md` step 0 for the audit snippet), and validate `vllm.sh 2` (the cute-DSL warmup path) before promoting.
