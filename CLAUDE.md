# vLLM Custom Docker Build

This directory is solely for building the base vLLM Docker image. It does **not** run models directly. Models are launched via `~/scripts/vllm.sh`, which uses this image as its base.

The image targets an RTX 5090 (Blackwell, compute 12.0) with cutting-edge CUDA drivers.

## Why a custom build?

> **NOTE (2026-07-02):** The driver has since been upgraded to **610.43.02 (CUDA 13.3 runtime)** — see Hardware table. The rationale below was written for the **old driver 580.x (CUDA 13.0 limit)** and its constraints no longer strictly apply: driver 610 runs CUDA 13.0/13.1/13.3 PTX, so the nvcc 13.0 override and the `26.03` base pin may now be **unnecessary**. The custom build strategy has **not** yet been re-evaluated against driver 610; treat the reasoning below as historical until it is. (The current build still works — nvcc 13.0 PTX is forward-compatible with driver 610.)

The official `vllm/vllm-openai:latest` image uses CUDA 12.9, which is incompatible with NVIDIA driver 580.x. The standard PyTorch cu130 triggers CUDA forward-compat error 803 on the RTX 5090. A custom build is needed to reconcile:

- **Driver 580.x** requires CUDA 13.1 runtime (only available in NVIDIA's internal PyTorch)
- **vLLM's CUDA extensions** require full CUDA dev headers not present in the NVIDIA container
- **nvcc 13.1** generates PTX that driver 580.x cannot run — so we substitute the CUDA 13.0 compiler

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
| `Dockerfile` | Multi-stage Dockerfile (see below) |
| `vllm/` | vLLM source repo (cloned) |
| `build.number` | Persisted monotonic build counter; `build.sh` increments it (on success only) and tags the image `vllm-custom:b<N>` |
| `build.history` | Append-only log of successful builds: build number, date/time, and vLLM version. Distinct from `build.log` (full build output) |

## Dockerfile strategy (3 stages)

1. **`cuda-devel`** — pulls `nvidia/cuda:13.0.1-devel-ubuntu22.04` for its nvcc 13.0, headers, and libdevice
2. **`builder`** — starts from `nvcr.io/nvidia/vllm:26.03.post1-py3` (NVIDIA's PyTorch with CUDA 13.1 runtime), then **overwrites** the CUDA 13.1 compiler toolchain with the 13.0 one from stage 0, builds vLLM from source into a wheel
3. **Runtime** — fresh `nvcr.io/nvidia/vllm:26.03.post1-py3` base, installs the wheel built in stage 2

## Transformers pin (load-bearing) & Gemma 4 MTP

The runtime stage installs the vLLM wheel with `--no-deps` (to preserve NVIDIA's CUDA 13.1 torch), so **transformers is never pulled as a wheel dependency** — the version is whatever the explicit `RUN pip install transformers==<X>` line sets. Removing that line leaves the base image's transformers **4.57.5**, which is 4.x and doesn't recognize the `gemma4` architecture at all → every Gemma 4 config breaks at model load. The pin is required for all Gemma 4 options, not optional.

**Pinned to `transformers==5.12.1`** (bumped from 5.5.4 on 2026-07-02) to enable Gemma 4 MTP:

- Gemma 4 MTP (Multi-Token Prediction) speculative decoding uses Google's drafter `google/gemma-4-31B-it-assistant` (~0.5B), enabled by `vllm.sh` config **`4m`** (same NVFP4 checkpoint as `4`, but with `--speculative-config` and context trimmed to **125k** so the drafter weights fit the KV budget at gpu-mem-util 0.983).
- The drafter's `config.json` has `model_type: gemma4_assistant`. vLLM does **not** register this in its config registry (true even on upstream main) — it relies on **transformers** to recognize it, and that model_type only landed in **transformers 5.12.x**. `5.7.0` does NOT have it (despite the drafter stamping `5.7.0.dev0`); with anything <5.12 the engine crashes at config load: "Transformers does not recognize `gemma4_assistant`".
- Validated 2026-07-02: 5.12.1 boots cleanly against our wheel, preserves NVIDIA's torch, and drives the drafter at ~80–91% draft acceptance (mean accept length ~1.8 → near ~1.8× decode). MTP is text-only (drafter has no vision tower); image prompts still run on the target.

**A newer base image does NOT fix this.** As of 2026-07 the newest NVIDIA base is `26.06-py3` (torch 2.13, CUDA 13.3, vLLM 0.22.1) but it still ships **transformers 5.6.0** — below 5.12.1, so the explicit pin is still needed. Bumping the base is also risky: the CUDA 13.0 nvcc override, `cuda_view.cu` patch, and `register_opaque_type` hoist patch are all tuned to `26.03`'s torch 2.11 / CUDA 13.1 and would need re-validating against driver 580.105.08. Keep the current base + explicit transformers pin.

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

## Planned: base image bump to `26.06-py3` (deferred)

Now that the driver is **610.43.02 (CUDA 13.3)**, the driver-580.x workarounds are obsolete and the base can move to `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13, CUDA 13.3, NVIDIA vLLM 0.22.1). This is a **maintainability/modernization** play — it is NOT required for Gemma 4 MTP (the transformers pin handles that on any base). **Do this as a separate, deliberate build effort — not bundled with unrelated changes.**

**Prereq:** let the current `transformers==5.12.1` build finish and validate `vllm.sh 4m` first, so there's a known-good baseline to diff against.

**Outline:**
1. Bump **both** `FROM` lines (builder + runtime) to `26.06-py3`.
2. **Delete the `cuda-devel` stage and the nvcc-13.0 override COPYs** — pure driver-580.x workarounds; on driver 610 use the base's native nvcc 13.3. (Overwriting a CUDA 13.3 base with 13.0 headers/nvcc is inconsistent.) This is the main simplification payoff.
3. **Keep** `RUN pip install transformers==5.12.1` — `26.06` ships only 5.6.0 (< 5.12.1, no `gemma4_assistant`).
4. Re-validate the torch/CUDA-keyed patches against torch 2.13; their grep guards fail-fast if stale:
   - `cuda_view.cu` patch (highest risk) — keyed to torch 2.11.0a0's stable ABI; torch 2.13 likely matches upstream, so **probably revert to upstream's file** rather than patch.
   - `register_opaque_type` hoist sed — may be unnecessary if 2.13 has the `hoist` param.
   - FA4 `VLLM_FLASH_ATTN_SRC_DIR` unset — re-check the base's flash-attn snapshot path.
5. Re-check runtime dep pins (`compressed-tensors==0.17.0`, `humming-kernels`, `tokenspeed-mla`, `tilelang`, `fastsafetensors`) against `26.06`'s 0.22.1 baseline — some may already be satisfied or now conflict.
6. Budget for 1–2 failed builds (~60 min each). Keep the last good `vllm-custom:b<N>` as rollback; `vllm.sh` tracks `:latest`, so a bad build only bites once retagged.

Consider trialing in a separate `Dockerfile.26.06` first to avoid disturbing the working `Dockerfile`.

## issue.md — Bug Investigation Log

`issue.md` in this directory is the **living investigation log** for a vLLM engine livelock bug (requests accumulate in waiting queue; engine never schedules them; no errors emitted).

**Rules for Claude:**
- **Always read `issue.md` at the start of any debugging session** for this bug to avoid re-treading covered ground.
- **Always update `issue.md` before reporting findings** — write results there first, then summarize to the user.
- **Append new investigation rounds** under a dated `## Round N — YYYY-MM-DD` heading; never overwrite prior rounds.
- Sections to maintain: Hypotheses (ordered by confidence), What We Have Not Yet Tried, and a Findings table per round.
- The file is the single source of truth; keep it complete enough that a fresh Claude session can pick up the investigation without this conversation.
