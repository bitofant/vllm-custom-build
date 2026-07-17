# vLLM Custom Docker Build

This directory is solely for building the base vLLM Docker image. It does **not** run models directly. Models are launched via `~/scripts/vllm.sh`, which uses this image as its base.

The image targets an RTX 5090 (Blackwell, compute 12.0) with cutting-edge CUDA drivers.

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
| `build.number` | Persisted monotonic build counter; `build.sh` increments it (on success only) and tags the image `vllm-custom:b<N>` |
| `build.history` | Append-only log of successful builds: build number, date/time, and vLLM version. Distinct from `build.log` (full build output) |

## Dockerfile strategy (2 stages, base `26.06-py3`)

1. **`builder`** — starts from `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13, CUDA 13.3) and builds vLLM from current source into a wheel using the base's **native nvcc 13.3**.
2. **Runtime** — fresh `nvcr.io/nvidia/vllm:26.06-py3` base, installs the wheel (`--no-deps` to preserve NVIDIA's custom torch), then pins the handful of runtime deps that current source needs newer than the base's vLLM 0.22.1.

> **26.03 → 26.06 bump (2026-07-17):** the old `cuda-devel` stage + nvcc-13.0 override, the `register_opaque_type` hoist sed, and the `cuda_view.cu` patch were all **deleted** — verified unnecessary on torch 2.13 / driver 610. The previous 3-stage 26.03 Dockerfile is archived as `Dockerfile.26.03`; see `PLAN-base-26.06.md` for the measured dep audit. Validated by building b14 and serving `vllm.sh 2` (Qwen NVFP4 MoE + MTP) end-to-end.

### Pins the runtime stage still needs (source ahead of 26.06's vLLM 0.22.1)
Because we build bleeding-edge source on a base tuned for 0.22.1, these stay pinned (guards fail-fast if a symbol/torch regresses): `transformers==5.12.1`, `xgrammar==0.2.3`, `flashinfer-python/-cubin==0.6.14` (+`--extra-index-url https://flashinfer.ai/whl/`), `compressed-tensors==0.17.0`, `humming-kernels==0.1.10`, `tokenspeed-mla==0.1.8`, `tilelang==0.1.9`, `nvidia-cutlass-dsl==4.6.0` + `quack-kernels==0.6.1` (matched cute-DSL pair), `apache-tvm-ffi==0.1.10`, `nvidia-cudnn-frontend>=1.19.1`, `mistral_common>=1.11.5`. The base bump **dropped** the previously-needed `openai` and `fastsafetensors` pins (26.06 satisfies them natively).

## Transformers pin (load-bearing) & Gemma 4 MTP

The runtime stage installs the vLLM wheel with `--no-deps` (to preserve NVIDIA's CUDA 13.1 torch), so **transformers is never pulled as a wheel dependency** — the version is whatever the explicit `RUN pip install transformers==<X>` line sets. Removing that line leaves the base image's transformers **4.57.5**, which is 4.x and doesn't recognize the `gemma4` architecture at all → every Gemma 4 config breaks at model load. The pin is required for all Gemma 4 options, not optional.

**Pinned to `transformers==5.12.1`** (bumped from 5.5.4 on 2026-07-02) to enable Gemma 4 MTP:

- Gemma 4 MTP (Multi-Token Prediction) speculative decoding uses Google's drafter `google/gemma-4-31B-it-assistant` (~0.5B), enabled by `vllm.sh` config **`4m`** (same NVFP4 checkpoint as `4`, but with `--speculative-config` and context trimmed to **125k** so the drafter weights fit the KV budget at gpu-mem-util 0.983).
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
2. No tag edit needed — `~/scripts/vllm.sh` is pinned to `vllm-custom:latest`, which always tracks the newest build. (Pin to a specific `b<N>` tag only if you need to roll back.)
3. Test: `source ~/scripts/vllm.sh; vllm rec 4` (removes and re-creates the vLLM docker container for Gemma4:31b)

## Base image bump to `26.06-py3` (DONE 2026-07-17, b14)

The base was moved from `26.03.post1` to `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13, CUDA 13.3, NVIDIA vLLM 0.22.1) on driver 610. Outcome:

- Deleted the `cuda-devel` stage, nvcc-13.0 override, `register_opaque_type` hoist sed, and `cuda_view.cu` patch — all verified unnecessary on torch 2.13 (register_opaque_type has `hoist` natively; `cuda_view.cu` compiles unpatched against 2.13's stable ABI).
- Dropped the `openai==2.45.0` and `fastsafetensors` pins (26.06 satisfies them).
- Kept `transformers==5.12.1` and the CUDA/kernel pins listed under "Dockerfile strategy".
- Fixed the cutlass/quack `ThrMma` crash via the matched `nvidia-cutlass-dsl==4.6.0` + `quack-kernels==0.6.1` pair.

Validated: b14 serves `vllm.sh 2` (Qwen NVFP4 MoE + MTP) end-to-end. `Dockerfile.26.03` archives the previous build; `PLAN-base-26.06.md` has the full audit/rationale.

> **Reminder for the next base bump:** we track bleeding-edge vLLM source, so *any* fixed NVIDIA base will lag some deps — expect to keep a set of CUDA/kernel pins in the runtime stage no matter what. Trial in a separate `Dockerfile.<ver>`, diff shipped versions against `vllm/requirements/{common,cuda}.txt` (see `PLAN-base-26.06.md` step 0 for the audit snippet), and validate `vllm.sh 2` (the cute-DSL warmup path) before promoting.

## issue.md — Bug Investigation Log

`issue.md` in this directory is the **living investigation log** for a vLLM engine livelock bug (requests accumulate in waiting queue; engine never schedules them; no errors emitted).

**Rules for Claude:**
- **Always read `issue.md` at the start of any debugging session** for this bug to avoid re-treading covered ground.
- **Always update `issue.md` before reporting findings** — write results there first, then summarize to the user.
- **Append new investigation rounds** under a dated `## Round N — YYYY-MM-DD` heading; never overwrite prior rounds.
- Sections to maintain: Hypotheses (ordered by confidence), What We Have Not Yet Tried, and a Findings table per round.
- The file is the single source of truth; keep it complete enough that a fresh Claude session can pick up the investigation without this conversation.
