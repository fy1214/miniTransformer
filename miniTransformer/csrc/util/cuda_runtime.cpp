#include "cuda_runtime.h"

#include <filesystem>
#include <mutex>

#include "logging.h"

namespace cuda {

int num_devices() {
  auto query_num_devices = []() -> int {
    int count;
    NVTE_CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
  };
  static int num_devices_ = query_num_devices();
  return num_devices_;
}

int sm_arch(int device_id) {
  static std::vector<int> cache(num_devices(), -1);
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&]() {
    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    cache[device_id] = 10 * prop.major + prop.minor;
  };
  std::call_once(flags[device_id], init);
  return cache[device_id];
}

int sm_count(int device_id) {
  static std::vector<int> cache(num_devices(), -1);
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&]() {
    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    cache[device_id] = prop.multiProcessorCount;
  };
  std::call_once(flags[device_id], init);
  return cache[device_id];
}

}  // namespace cuda
