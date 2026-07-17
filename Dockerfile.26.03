# Build vLLM from source on top of NVIDIA's container.
#
# NVIDIA's container provides:
#   - CUDA 13.1 PyTorch (nvinternal build) which works with driver 580.x
#   - nvcc compiler
#
# The standard CUDA devel image provides:
#   - Full CUDA development headers (cusparse.h, cublas.h, etc.)
#   - Required to compile vLLM's CUDA extensions
#
# This two-base approach is necessary because:
#   - vllm/vllm-openai:latest uses CUDA 12.9 (incompatible with driver 580.x)
#   - Standard PyTorch cu130 triggers CUDA forward compat error 803 on RTX 5090
#   - Driver 580.x requires CUDA 13.1 runtime (only in NVIDIA internal PyTorch)
#   - NVIDIA container lacks full CUDA dev headers needed to compile vLLM kernels

# Stage 0: Pull CUDA 13.0 toolkit (headers + nvcc) from standard devel image.
# Using 13.0 nvcc ensures PTX is compatible with driver 580.x (CUDA 13.0).
# Using 13.1 nvcc (from NVIDIA container) generates PTX that driver 580.x can't run.
FROM nvidia/cuda:13.0.1-devel-ubuntu22.04 AS cuda-devel

# Stage 1: Build vLLM using NVIDIA's PyTorch + CUDA 13.0 nvcc + CUDA 13.0 headers
FROM nvcr.io/nvidia/vllm:26.03.post1-py3 AS builder

ARG TORCH_CUDA_ARCH_LIST="12.0"
ARG MAX_JOBS=16
ARG NVCC_THREADS=4

# Override NVIDIA container's CUDA 13.1 compiler with CUDA 13.0 compiler.
# This replaces nvcc, ptxas, cicc, and libdevice so all PTX generated is
# compatible with driver 580.x (CUDA 13.0). PyTorch's CUDA 13.1 runtime
# (for GPU init) is unaffected - it uses its own bundled libraries.
COPY --from=cuda-devel /usr/local/cuda/include /usr/local/cuda/include
COPY --from=cuda-devel /usr/local/cuda/bin /usr/local/cuda/bin
COPY --from=cuda-devel /usr/local/cuda/nvvm /usr/local/cuda/nvvm

WORKDIR /workspace/vllm-build

# Remove old vLLM (keep PyTorch and all other CUDA libs)
RUN pip uninstall -y vllm vllm-flash-attn flashinfer-cubin flashinfer-jit-cache 2>/dev/null || true

# Install build tools. setuptools_rust is needed by vllm's setup.py since the
# Rust frontend integration (PR #40848, May 2026). The RustExtension itself is
# optional unless VLLM_REQUIRE_RUST_FRONTEND is set, so if cargo is missing the
# rust binary build is skipped with a warning — but setup.py still imports
# setuptools_rust at module load time, so it must be present.
RUN pip install --no-cache-dir setuptools wheel cmake ninja packaging setuptools_scm setuptools_rust

# Copy source
COPY vllm/ /workspace/vllm-build/

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV VLLM_DOCKER_BUILD_CONTEXT=1

# Drop `hoist=True` kwarg from register_opaque_type call. NVIDIA's torch
# 2.11.0a0+nv26.03 has register_opaque_type but without the `hoist` parameter
# (added in a later torch). `hoist=True` is a torch.compile optimization that
# avoids per-layer MoE recompilation; removing it is a perf cost only, not a
# correctness issue. Needed to run this vLLM source on NVIDIA's base.
RUN sed -i 's/register_opaque_type(LayerName, typ="value", hoist=True)/register_opaque_type(LayerName, typ="value")/' \
    vllm/utils/torch_utils.py && \
    grep -q 'register_opaque_type(LayerName, typ="value")' vllm/utils/torch_utils.py

# NOTE: The Gemma4 streaming-reasoning patch was removed as of vLLM
# v0.23.1rc0 (upstream PR #45588, "Replace legacy Gemma4 parsers with
# engine-based implementation"). The old BaseThinkingReasoningParser
# delegation — which required <|channel> to appear in the stream and so
# leaked the chain of thought into `content` when the chat template
# pre-emitted the open token — is gone. The new engine-based Gemma4Parser
# (vllm/parser/gemma4.py) fixes this directly: _preprocess_feed injects
# CHANNEL_START on the first delta when the prompt pre-emitted it, and
# adjust_initial_state_from_prompt pre-initialises the engine to REASONING.
# gemma4_reasoning_parser.patched.py is kept in this dir for reference only.

# Overwrite csrc/libtorch_stable/cuda_view.cu. Upstream commit 56aff0dd15c
# ("[10/n] Migrate cuda_view ... to torch stable ABI") targets a newer
# torch::stable API than NVIDIA's torch 2.11.0a0 base ships:
#   - torch::stable::Tensor has no .layout() member here
#   - from_blob's deleter is a plain DeleterFnPtr (void(*)(void*)), but upstream
#     passes capturing lambdas, which don't convert
# Our patched version forwards std::nullopt for layout and routes the per-view
# cleanup through a mutex-guarded registry + non-capturing trampoline. The grep
# guard fires if upstream changes this file (e.g. drops the capturing lambda)
# so we know to revisit the patch. See issue: 3 errors in cuda_view.cu.o build.
COPY cuda_view.patched.cu /tmp/cuda_view.patched.cu
RUN grep -q 'keep cpu tensor alive' csrc/libtorch_stable/cuda_view.cu && \
    cp /tmp/cuda_view.patched.cu csrc/libtorch_stable/cuda_view.cu && \
    grep -q 'CUDA_VIEW_PATCHED_FOR_NV2611' csrc/libtorch_stable/cuda_view.cu

# Pre-create vllm_flash_attn/cute package dir so find_packages() discovers it
# at setup() time — find_packages runs before cmake install populates cute/.
RUN mkdir -p vllm/vllm_flash_attn/cute && touch vllm/vllm_flash_attn/cute/__init__.py

# Build vLLM wheel against NVIDIA's CUDA 13.1 PyTorch.
# Unset VLLM_FLASH_ATTN_SRC_DIR: NVIDIA's base image points it at
# /opt/vllm/vllm-flash-attn/, an older flash-attention snapshot that predates
# FA4 and lacks flash_attn/cute/. With it set, vllm's cmake treats it as "local
# dev mode" and installs the FA4 cutedsl files as a SYMLINK to that missing
# path, which bdist_wheel then trips over ("can't copy ... cute: doesn't exist
# or not a regular file"). Unsetting forces FetchContent to clone the pinned
# flash-attention commit fresh and the else() branch copies real cute/*.py
# files into the build tree.
RUN env -u VLLM_FLASH_ATTN_SRC_DIR python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38


# Stage 2: Runtime image - NVIDIA base + newly built vLLM
FROM nvcr.io/nvidia/vllm:26.03.post1-py3

# Remove old vLLM
RUN pip uninstall -y vllm vllm-flash-attn flashinfer-cubin flashinfer-jit-cache 2>/dev/null || true

# Install the freshly built vLLM wheel.
# --no-deps is critical: our wheel pins torch==2.10.0, and without this pip
# would pull cu128 torch from PyPI and clobber NVIDIA's cu13.2 torch that the
# wheel was actually compiled against — producing a c10 symbol mismatch at
# import time. NVIDIA's base already has vLLM 0.17.1's runtime deps installed;
# we only uninstall `vllm` itself above, so --no-deps leaves them in place.
COPY --from=builder /workspace/vllm-build/dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm /tmp/*.whl

# Upgrade transformers for gemma4 support. The NVIDIA base ships transformers
# 4.57.5 which doesn't recognize the gemma4 architecture at all.
#
# Pinned to 5.12.1 (not 5.5.4) because Gemma 4 MTP speculative decoding needs it:
# the drafter google/gemma-4-31B-it-assistant declares model_type
# `gemma4_assistant`, which vLLM does NOT register in its own config registry
# (true even on upstream main) — it relies on transformers to recognize that
# model_type, and it was only added in transformers 5.12.x. With 5.5.4 the MTP
# config crashes at load ("Transformers does not recognize gemma4_assistant").
# 5.12.1 verified 2026-07-02 to boot cleanly against our vllm wheel, preserve
# NVIDIA's torch, and drive the gemma4 MTP drafter (~80-91% draft acceptance).
# See vllm.sh config `4m`.
RUN pip install --no-cache-dir transformers==5.12.1

# Install runtime deps that vLLM main added since 0.17.1 (NVIDIA's base
# version). Our wheel install used --no-deps to preserve NVIDIA's custom
# torch, so these need explicit installs. `humming-kernels` is the
# load-bearing one — vllm/model_executor/layers/quantization/__init__.py
# eagerly imports it from get_quantization_config(), so ANY model with
# a quantization config (gemma4 NVFP4 included) crashes on startup if
# the `humming` package is missing. `compressed-tensors` is the same kind
# of trap: get_quantization_config("compressed-tensors") eagerly imports
# `compressed_tensors.compressors.pack_quantized`, which only exists in
# 0.17.0 (the NVIDIA base ships 0.13.0) — every NVFP4/compressed-tensors
# model crashes on startup without the bump. The others are pinned to match
# requirements/cuda.txt; compressed-tensors==0.17.0 matches common.txt.
# Pin to the exact versions the wheel requires so this stays in lockstep.
RUN pip install --no-cache-dir \
    'humming-kernels[cu13]==0.1.10' \
    tokenspeed-mla==0.1.8 \
    tilelang==0.1.9 \
    fastsafetensors==0.3.2 \
    compressed-tensors==0.17.0

# Upgrade xgrammar to the version vLLM main is built against.
#
# The NVIDIA base ships xgrammar 0.1.32, but vLLM main
# (requirements/common.txt) pins `xgrammar >= 0.2.1, < 1.0.0`. Because our
# wheel install used --no-deps to preserve NVIDIA's custom torch, the old
# 0.1.32 is left in place. vLLM's tool-parser code now does
#   from xgrammar import StructuralTag, normalize_tool_choice
# (vllm/tool_parsers/structural_tag_registry.py) and normalize_tool_choice only
# landed in xgrammar 0.2.1 — 0.1.32 has StructuralTag but NOT
# normalize_tool_choice. Result: any request that touches tool-choice /
# structured output fails the server startup import with HTTP 500
#   ImportError: cannot import name 'normalize_tool_choice' from 'xgrammar'
#
# 0.2.3 is the newest release satisfying vLLM's `< 1.0.0` bound and exports both
# symbols at top level. Its torch dep is unpinned (torch>=1.10.0), already
# satisfied by NVIDIA's torch 2.11, so pip leaves torch in place; the only new
# transitive dep is apache-tvm-ffi (additive). The guards below fail the build
# fast if the symbol goes missing again or if NVIDIA's torch gets clobbered.
RUN pip install --no-cache-dir xgrammar==0.2.3 && \
    python3 -c "from xgrammar import StructuralTag, normalize_tool_choice" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# Bump openai for the tool-parser import chain (NamespaceTool). The NVIDIA base
# ships openai 2.24.0. vLLM main's requirements only FLOOR it at `openai >= 2.0.0`
# (requirements/common.txt) — an under-specified pin: vllm/tool_parsers/utils.py
# does `from openai.types.responses import NamespaceTool`, and NamespaceTool only
# landed in openai 2.25.0. So 2.24.0 satisfies the declared floor yet crashes the
# server at STARTUP (import of tool_parsers, pulled in by cli_args → api_server),
# crash-looping EVERY model — not just tool-use ones — with:
#   ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'
# Our --no-deps wheel install leaves the base's 2.24.0 in place, so pin here.
# 2.45.0 = latest stable; openai is pure-python with loose deps (httpx/pydantic
# already satisfied), so this touches neither NVIDIA's torch nor anything CUDA.
# Validated 2026-07-17: openai 2.45.0 in vllm-custom:b12 imports tool_parsers +
# cli_args + NamespaceTool cleanly. Guard fail-fasts if the symbol/torch regress.
RUN pip install --no-cache-dir openai==2.45.0 && \
    python3 -c "from openai.types.responses import NamespaceTool" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# Upgrade FlashInfer to the version vLLM main is built against.
#
# The NVIDIA base ships an internal FlashInfer 0.6.7 (0.6.7+...nvinternal.cu132).
# vLLM main (requirements/cuda.txt) pins flashinfer-python==0.6.14 /
# flashinfer-cubin==0.6.14, and its fp8 KV-cache FlashInfer prefill path calls
#
# INDEX NOTE (2026-07-16): since 0.6.14, flashinfer-cubin is NO LONGER on PyPI —
# it's published only on flashinfer's own index. A bare `pip install
# flashinfer-cubin==0.6.14` fails with "No matching distribution" (PyPI tops out
# at cubin 0.6.13). We mirror cuda.txt's own directive and add
# `--extra-index-url https://flashinfer.ai/whl/`, which serves the
# flashinfer_cubin-0.6.14-py3-none-any.whl. (Upstream also drops cubin from the
# wheel's install_requires via setup.py so the published wheel carries no
# unresolvable pin; our --no-deps wheel install means we must add it explicitly
# here anyway.) flashinfer-python 0.6.14 remains on PyPI. Both are pure
# py3-none-any wheels, so this changes no ABI. Original rationale:
# BatchPrefillWithPagedKVCacheWrapper.run(kv_cache_sf=...). 0.6.7 predates that
# kwarg, so NVFP4 checkpoints whose KV cache resolves to fp8 (e.g.
# unsloth/Qwen3.6-35B-A3B-NVFP4 under --kv-cache-dtype auto, which MTP
# spec-decode also forces onto the FLASHINFER backend) crash the EngineCore on
# the first request with:
#   TypeError: ...run() got an unexpected keyword argument 'kv_cache_sf'
#
# flashinfer-python is a pure-python (py3-none-any) JIT wheel — it compiles
# kernels at runtime with the base's nvcc 13.2 — so this bump does NOT rebuild
# anything and does NOT touch NVIDIA's custom torch 2.11 (verified: torch's dep
# is unpinned and already satisfied, so pip leaves it in place; the only other
# packages pulled are flashinfer's CUDA-JIT toolchain, all additive). sm120 /
# NVFP4 (compressed-tensors) support is retained. Validated 2026-07-16 against
# vllm-custom:b8 (vllm 0.23.1rc1.dev791): boots, run() accepts kv_cache_sf.
#
# The grep-style guard below fail-fasts if a future base/vLLM bump ships a
# FlashInfer whose run() lost the kwarg again — the exact regression this fixes.
RUN pip install --no-cache-dir --extra-index-url https://flashinfer.ai/whl/ \
        flashinfer-python==0.6.14 flashinfer-cubin==0.6.14 && \
    python3 -c "import inspect, flashinfer; from flashinfer import BatchPrefillWithPagedKVCacheWrapper as W; assert 'kv_cache_sf' in inspect.signature(W.run).parameters, 'FlashInfer run() is missing kv_cache_sf — version skew with vLLM main'" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# Copy runtime files
COPY --from=builder /workspace/vllm-build/examples /workspace/examples
COPY --from=builder /workspace/vllm-build/benchmarks /workspace/benchmarks

ENV VLLM_USAGE_SOURCE=production-docker-image

ENTRYPOINT ["vllm", "serve"]
