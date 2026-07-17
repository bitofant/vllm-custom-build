# vLLM from source on NVIDIA's 26.06-py3 base (torch 2.13, CUDA 13.3, NVIDIA
# vLLM 0.22.1). Driver 610.43.02 (CUDA 13.3) runs the base's native nvcc 13.3, so
# the driver-580.x workarounds from the old 26.03 build are gone: no cuda-devel
# stage, no nvcc-13.0 override, no hoist sed, no cuda_view.cu patch.
#
# Promoted from Dockerfile.26.06 on 2026-07-17 after the trial build passed the
# full dep audit + import guards. See PLAN-base-26.06.md for rationale/audit and
# Dockerfile.26.03 for the archived previous (26.03) build.

# Stage 1: Build the vLLM wheel against the base's native torch 2.13 + nvcc 13.3.
FROM nvcr.io/nvidia/vllm:26.06-py3 AS builder

ARG TORCH_CUDA_ARCH_LIST="12.0"
ARG MAX_JOBS=16
ARG NVCC_THREADS=4

WORKDIR /workspace/vllm-build

# Remove old vLLM (keep PyTorch and all other CUDA libs)
RUN pip uninstall -y vllm vllm-flash-attn flashinfer-cubin flashinfer-jit-cache 2>/dev/null || true

# Build tools (setuptools_rust imported at setup.py module load; see 26.03 notes).
RUN pip install --no-cache-dir setuptools wheel cmake ninja packaging setuptools_scm setuptools_rust

COPY vllm/ /workspace/vllm-build/

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}
ENV NVCC_THREADS=${NVCC_THREADS}
ENV VLLM_DOCKER_BUILD_CONTEXT=1

# DROPPED vs 26.03 (both verified unnecessary on torch 2.13):
#  - register_opaque_type `hoist=True` sed strip: torch 2.13's
#    register_opaque_type already accepts `hoist` (verified 2026-07-17), so the
#    MoE recompile optimization is kept natively — no strip needed.
#  - cuda_view.cu patch: was keyed to torch 2.11.0a0's older torch::stable ABI
#    (no .layout(), plain DeleterFnPtr). torch 2.13 is expected to match
#    upstream commit 56aff0dd's newer stable API. IF the build fails on
#    csrc/libtorch_stable/cuda_view.cu, re-add the guarded cuda_view.patched.cu
#    COPY from the 26.03 Dockerfile.

# Pre-create vllm_flash_attn/cute so find_packages() discovers it at setup() time.
RUN mkdir -p vllm/vllm_flash_attn/cute && touch vllm/vllm_flash_attn/cute/__init__.py

# Unset VLLM_FLASH_ATTN_SRC_DIR so cmake clones flash-attention fresh instead of
# symlinking to the base's older snapshot (bdist_wheel trips on the dangling cute
# symlink otherwise). Re-verify the base's snapshot path if this ever fails.
RUN env -u VLLM_FLASH_ATTN_SRC_DIR python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38


# Stage 2: Runtime — 26.06 base + freshly built wheel.
FROM nvcr.io/nvidia/vllm:26.06-py3

RUN pip uninstall -y vllm vllm-flash-attn flashinfer-cubin flashinfer-jit-cache 2>/dev/null || true

# --no-deps preserves NVIDIA's custom torch 2.13 (the wheel's torch pin would
# otherwise pull stock torch and cause a c10 ABI mismatch).
COPY --from=builder /workspace/vllm-build/dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm /tmp/*.whl

# transformers: 26.06 ships 5.6.0; keep 5.12.1 for Gemma 4 MTP (drafter
# model_type `gemma4_assistant`, only recognized by transformers 5.12.x). See 4m.
RUN pip install --no-cache-dir transformers==5.12.1

# Kernel/quant runtime deps that source 1214 needs NEWER than 26.06's vLLM 0.22.1
# baseline (measured in PLAN-base-26.06.md). --no-deps wheel install left the
# base's older versions in place, so pin explicitly to match requirements/*.txt.
RUN pip install --no-cache-dir \
    'humming-kernels[cu13]==0.1.10' \
    tokenspeed-mla==0.1.8 \
    tilelang==0.1.9 \
    compressed-tensors==0.17.0 \
    apache-tvm-ffi==0.1.10 \
    'nvidia-cudnn-frontend>=1.19.1' \
    'mistral_common[image]>=1.11.5'

# xgrammar: 26.06 ships 0.2.0 (< 0.2.1, lacks normalize_tool_choice). Keep 0.2.3.
RUN pip install --no-cache-dir xgrammar==0.2.3 && \
    python3 -c "from xgrammar import StructuralTag, normalize_tool_choice" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# cute-DSL FA4 stack (the 26.03 blocker fix, cleaner here). 26.06 ships
# cutlass-dsl 4.5.2 + quack 0.4.1; source 1214 pins nvidia-cutlass-dsl==4.6.0.
# quack 0.6.1 declares cutlass-dsl==4.6.0 and its layout_utils no longer
# references the removed cutlass.cute.core.ThrMma symbol — verified import-clean
# on 26.06 (2026-07-17). The guard imports the exact modules vLLM's cute-DSL
# kernel warmup pulls in (quack.compile_utils / quack.layout_utils), which is
# what crash-looped EngineCore on the mismatched 26.03 pair.
RUN pip install --no-cache-dir 'nvidia-cutlass-dsl[cu13]==4.6.0' quack-kernels==0.6.1 && \
    python3 -c "from quack.compile_utils import make_fake_tensor; import quack.layout_utils" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# flashinfer: 26.06 ships 0.6.12; source 1214 pins 0.6.14. cubin left PyPI at
# 0.6.14 → needs flashinfer's own index (--extra-index-url). Guard checks the
# kv_cache_sf kwarg (the fp8-KV NVFP4 regression) is present.
RUN pip install --no-cache-dir --extra-index-url https://flashinfer.ai/whl/ \
        flashinfer-python==0.6.14 flashinfer-cubin==0.6.14 && \
    python3 -c "import inspect, flashinfer; from flashinfer import BatchPrefillWithPagedKVCacheWrapper as W; assert 'kv_cache_sf' in inspect.signature(W.run).parameters, 'FlashInfer run() is missing kv_cache_sf — version skew with vLLM main'" && \
    python3 -c "import torch; assert 'nv' in torch.__version__, f'NVIDIA torch got clobbered: {torch.__version__}'"

# openai: NO pin needed — 26.06 ships 2.44.0, which has NamespaceTool (added in
# 2.25.0). The 26.03 Dockerfile pinned openai==2.45.0 for exactly this symbol;
# the base bump makes that pin obsolete.

COPY --from=builder /workspace/vllm-build/examples /workspace/examples
COPY --from=builder /workspace/vllm-build/benchmarks /workspace/benchmarks

ENV VLLM_USAGE_SOURCE=production-docker-image
ENTRYPOINT ["vllm", "serve"]
