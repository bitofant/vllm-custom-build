// PATCHED for NVIDIA base image torch 2.11.0a0 (nvcr.io/nvidia/vllm:26.03).
// Upstream vllm/csrc/libtorch_stable/cuda_view.cu (commit 56aff0dd15c,
// "[10/n] Migrate cuda_view ... to torch stable ABI") targets a newer
// torch::stable API than the base image ships. Two incompatibilities:
//
//   1. torch::stable::Tensor has no .layout() method here. It is only used to
//      forward the layout into torch::stable::empty(), whose `layout` argument
//      is optional, so we pass std::nullopt (Strided is the default anyway).
//
//   2. from_blob's deleter parameter is a plain function pointer
//      (DeleterFnPtr = void(*)(void*)), NOT a std::function. Upstream passes
//      capturing lambdas ([base = cpu_tensor] / [host_ptr]) which cannot
//      convert to a function pointer. We register the per-view cleanup in a
//      mutex-guarded multimap keyed by the data pointer and install a single
//      non-capturing trampoline (cuda_view_deleter) as the DeleterFnPtr.
//      A multimap (not map) is used so two live views sharing a data pointer
//      each keep their own cleanup entry; the trampoline erases exactly one.
//
// Guard marker for the Dockerfile grep assert: CUDA_VIEW_PATCHED_FOR_NV2611
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/version.h>
#include <cuda_runtime.h>

#include <array>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <utility>

namespace {

// Registry of per-view cleanup callbacks, keyed by the blob data pointer.
// A non-capturing DeleterFnPtr can carry no state, so we stash the (possibly
// capturing) cleanup here and recover it from the data pointer at delete time.
std::mutex& cuda_view_registry_mutex() {
  static std::mutex m;
  return m;
}

std::multimap<void*, std::function<void()>>& cuda_view_registry() {
  static std::multimap<void*, std::function<void()>> r;
  return r;
}

void register_cuda_view_cleanup(void* data_ptr, std::function<void()> fn) {
  std::lock_guard<std::mutex> lock(cuda_view_registry_mutex());
  cuda_view_registry().emplace(data_ptr, std::move(fn));
}

// Non-capturing trampoline installed as the from_blob DeleterFnPtr. Pops a
// single cleanup entry for this data pointer and runs it outside the lock.
void cuda_view_deleter(void* data_ptr) {
  std::function<void()> fn;
  {
    std::lock_guard<std::mutex> lock(cuda_view_registry_mutex());
    auto& registry = cuda_view_registry();
    auto it = registry.find(data_ptr);
    if (it != registry.end()) {
      fn = std::move(it->second);
      registry.erase(it);
    }
  }
  if (fn) {
    fn();
  }
}

}  // namespace

// This function assumes that `cpu_tensor` is a CPU tensor,
// and that UVA (Unified Virtual Addressing) is enabled.
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor) {
  STD_TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  const auto dtype = cpu_tensor.scalar_type();
  const torch::stable::Device cuda_dev(torch::headeronly::DeviceType::CUDA);

  // handle empty tensor
  if (cpu_tensor.numel() == 0) {
    return torch::stable::empty(cpu_tensor.sizes(), dtype, std::nullopt,
                                cuda_dev);
  }

  std::array<StableIValue, 2> is_pinned_stack{
      torch::stable::detail::from(cpu_tensor),
      torch::stable::detail::from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::is_pinned", "", is_pinned_stack.data(), TORCH_ABI_VERSION));
  if (torch::stable::detail::to<bool>(is_pinned_stack[0])) {
    // If CPU tensor is pinned, directly get the device pointer.
    void* host_ptr = const_cast<void*>(cpu_tensor.mutable_data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    STD_TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
                    cudaGetErrorString(err));

    // keep cpu tensor alive for as long as the view lives
    register_cuda_view_cleanup(device_ptr, [base = cpu_tensor]() {});
    return torch::stable::from_blob(device_ptr, cpu_tensor.sizes(),
                                    cpu_tensor.strides(), cuda_dev, dtype,
                                    cuda_view_deleter);
  }

  // If CPU tensor is not pinned, allocate a new pinned memory buffer.
  torch::stable::Tensor contiguous_cpu = torch::stable::contiguous(cpu_tensor);
  size_t nbytes = contiguous_cpu.numel() * contiguous_cpu.element_size();

  void* host_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    STD_TORCH_CHECK(false, "cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  err = cudaMemcpy(host_ptr, contiguous_cpu.const_data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(false, "cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr);
    STD_TORCH_CHECK(
        false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  register_cuda_view_cleanup(device_ptr,
                             [host_ptr]() { cudaFreeHost(host_ptr); });
  return torch::stable::from_blob(device_ptr, contiguous_cpu.sizes(),
                                  contiguous_cpu.strides(), cuda_dev,
                                  contiguous_cpu.scalar_type(),
                                  cuda_view_deleter);
}
