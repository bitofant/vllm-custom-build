# Plan: Base image bump 26.03 → 26.06-py3

Status: **proposed** (2026-07-17). Supersedes the deferred outline in `CLAUDE.md`.
Owner decision: bump the NVIDIA base rather than keep pinning deps piecemeal on 26.03.

## Why now (session findings, 2026-07-17)

Rolling vLLM forward 423 commits (v0.23.1rc0-791 → -1214) exposed that the
**26.03 base is broadly stale** vs current vLLM `requirements/{common,cuda}.txt`.
A full audit of `vllm-custom:b13` found **11 mismatched + 4 missing** deps. Fixes
so far were whack-a-mole band-aids:

- xgrammar 0.1.32 → 0.2.3 (tool-choice 500)
- openai 2.24.0 → 2.45.0 (`NamespaceTool` startup crash)
- flashinfer 0.6.13 → 0.6.14 (via `--extra-index-url https://flashinfer.ai/whl/`)
- humming-kernels/tokenspeed-mla/fastsafetensors bumped to cuda.txt pins

**Current hard blocker (unfixed):** base ships `nvidia-cutlass-dsl 4.6.1` +
`quack-kernels 0.3.9`; vLLM's cute-DSL warmup (`quack/layout_utils.py`) calls
`cutlass.cute.core.ThrMma`, removed in 4.6.1. This runs under
`has_device_capability(90)` — **unconditional on the RTX 5090 (sm_120)** — so
**every** config crash-loops at EngineCore init. There is therefore **no
known-good baseline on 26.03 with current source** to diff against; the last
fully-working image is b8 (old source 791). This removes the original prereq
("validate current baseline first") and argues for going straight to 26.06,
whose vLLM 0.22.1 ships a mutually-consistent cutlass-dsl/quack pair.

## Objective

Move builder + runtime `FROM` to `nvcr.io/nvidia/vllm:26.06-py3` (torch 2.13,
CUDA 13.3, NVIDIA vLLM 0.22.1), delete the driver-580.x-era workarounds, and
re-validate the still-needed pins + patches against torch 2.13. Driver is now
610.43.02 (CUDA 13.3), so the 13.0 nvcc override and 26.03 pin are obsolete.

## Execution steps

Trial in a **separate `Dockerfile.26.06`** first (don't disturb the working
`Dockerfile`); promote only after `vllm.sh 4m` + `2` validate.

0. **Inspect what 26.06 actually ships** (do this FIRST — drives every pin below):
   ```bash
   docker pull nvcr.io/nvidia/vllm:26.06-py3
   docker run --rm --entrypoint python3 nvcr.io/nvidia/vllm:26.06-py3 -c "
   import importlib.metadata as m
   for p in ['torch','transformers','openai','xgrammar','flashinfer-python',
     'flashinfer-cubin','nvidia-cutlass-dsl','quack-kernels','humming-kernels',
     'tokenspeed-mla','tilelang','fastsafetensors','compressed-tensors',
     'llguidance','outlines_core','mistral_common','nvidia-cudnn-frontend',
     'apache-tvm-ffi','starlette','prometheus-fastapi-instrumentator']:
       try: print(f'{p:28} {m.version(p)}')
       except Exception as e: print(f'{p:28} MISSING')
   "
   ```
   Then diff against `vllm/requirements/{common,cuda}.txt` (reuse the audit
   snippet from this session) and pin ONLY what's still behind.

1. **Bump both `FROM` lines** (builder + runtime) to `26.06-py3`.

2. **Delete the `cuda-devel` stage + nvcc-13.0 override COPYs.** Pure driver-580.x
   workarounds; on driver 610 use the base's native nvcc 13.3.

3. **Re-validate torch/CUDA-keyed patches against torch 2.13** (grep guards fail
   fast if stale):
   - `cuda_view.cu` patch — keyed to torch 2.11.0a0 ABI; **likely revert to
     upstream's file** on 2.13 (highest-risk item).
   - `register_opaque_type` hoist sed — may be unnecessary if 2.13 has `hoist`.
   - FA4 `VLLM_FLASH_ATTN_SRC_DIR` unset — re-check base's flash-attn snapshot path
     (builder must `env -u VLLM_FLASH_ATTN_SRC_DIR`; see memory note).

4. **Reconcile pins** — see table. Keep only genuinely-still-behind ones.

5. **Budget 1–2 failed builds (~60 min each).** Keep last good `vllm-custom:b<N>`
   as rollback; `vllm.sh` tracks `:latest`, so a bad build only bites once retagged.

## Step 0 RESULTS (measured 2026-07-17, 26.06-py3 vs current source 1214)

26.06 ships: torch **2.13.0a0+nv**, CUDA 13.3, vLLM **0.22.1**, transformers 5.6.0,
openai 2.44.0, xgrammar 0.2.0, flashinfer 0.6.12, **cutlass-dsl 4.5.2 + quack 0.4.1**,
humming 0.1.2, tokenspeed 0.1.2, compressed-tensors 0.15.0.1, starlette 1.3.1,
outlines_core 0.2.14, llguidance 1.7.6, prometheus-instrumentator 8.0.2,
mistral_common 1.11.4, cudnn-frontend 1.18.0, apache-tvm-ffi 0.1.7.

**Key takeaway:** 26.06 does NOT zero out pins, because we build **bleeding-edge
source 1214 on a base tuned for 0.22.1** — the base still lags several deps. But
it's a clear win: it removes the driver-580 workarounds AND lets us **drop ~6
pins** that 26.06 now satisfies natively. ~10 CUDA/kernel-dep pins remain.

### Pins we can DROP on 26.06 (now satisfied natively)
- `openai` (2.44.0 ≥ has `NamespaceTool`) — drop the 2.45.0 pin
- `fastsafetensors` (0.3.2) — drop
- `starlette` (1.3.1 ≥ 1.0.1), `outlines_core` (0.2.14), `llguidance` (1.7.6),
  `prometheus-fastapi-instrumentator` (8.0.2) — never needed on 26.06
  (these were the risky/latent 26.03 bumps — the base bump avoids them entirely)

### Pins we must KEEP / ADD (26.06 ships older than source 1214 needs)
| Dep | 26.06 ships | source 1214 needs | action |
|-----|-------------|-------------------|--------|
| transformers | 5.6.0 | (gemma4 MTP) | **keep 5.12.1** |
| xgrammar | 0.2.0 | ≥0.2.1 | keep 0.2.3 |
| flashinfer-python/-cubin | 0.6.12 | ==0.6.14 | keep 0.6.14 **+ `--extra-index-url https://flashinfer.ai/whl/`** |
| compressed-tensors | 0.15.0.1 | ==0.17.0 | keep 0.17.0 |
| humming-kernels | 0.1.2 | ==0.1.10 | keep 0.1.10 |
| tokenspeed-mla | 0.1.2 | ==0.1.8 | keep 0.1.8 |
| nvidia-cutlass-dsl + quack-kernels | 4.5.2 + 0.4.1 | ==4.6.0 + ≥0.4.0 | **pin 4.6.0 + quack 0.6.1** (0.6.1 pins cutlass==4.6.0; provides `ThrMma`) — fixes the current blocker |
| apache-tvm-ffi | 0.1.7 | ==0.1.10 | add pin 0.1.10 |
| nvidia-cudnn-frontend | 1.18.0 | ≥1.19.1 | add pin |
| mistral_common | 1.11.4 | ≥1.11.5 | add pin (tiny) |
| numba | 0.65.1 | ==0.65.0 | base AHEAD; likely ignore (0.65.1 fine) |

### Keep base's (NVIDIA stack — never pin, would clobber custom torch)
torch, torchvision, torchaudio (missing), torchcodec (missing), PyNvVideoCodec
(missing), nvtx (missing). Mismatch vs cuda.txt is expected/tolerated.

**Conclusion:** proceed with the bump. Net pin count drops and the driver-580
workarounds go, but the CUDA/kernel pins persist as long as we track bleeding-edge
source ahead of the base's vLLM — that's inherent to the build strategy, not a
26.06 deficiency.

## Risks & mitigations

- **cuda_view.cu ABI drift** (highest) — grep guard fails build; expect to revert
  to upstream file on torch 2.13.
- **cutlass/quack still skewed** — 26.06's base *should* be self-consistent; verify
  in step 0. If its vLLM 0.22.1 needs a different pair than base ships, pin to match.
- **No 26.03 baseline to diff** — validate behaviorally against b8 (old) + the
  `vllm.sh` smoke tests, not a byte-diff.
- **transformers 5.12.1 on torch 2.13** — re-confirm it still preserves base torch
  and drives the `4m` drafter.

## Validation checklist (before promoting Dockerfile.26.06 → Dockerfile)

- [ ] `docker images | grep vllm-custom` shows new build tagged
- [ ] `import torch` → version carries `nv` build tag (base torch preserved)
- [ ] Full dep audit (this session's snippet) → 0 unexpected mismatches
- [ ] `vllm rec 4 -y` (gemma dense) → healthy, chat returns choices
- [ ] `vllm rec 4m -y` (gemma MTP) → healthy, draft acceptance ~80–91%
- [ ] `vllm rec 2 -y` (Qwen NVFP4 MoE) → healthy, **passes cute-DSL warmup** (the
      ThrMma path), chat returns choices, no `kv_cache_sf` error
- [ ] image size ~15 GB, build time ~60 min

## Rollback

`vllm.sh` is pinned to `vllm-custom:latest`. If a 26.06 build regresses, retag the
last good build: `docker tag vllm-custom:b<good> vllm-custom:latest`. Keep
`Dockerfile` (26.03) untouched until 26.06 passes the full checklist.
