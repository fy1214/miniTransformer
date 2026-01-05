#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

using byte = uint8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
using fp8e8m0 = __nv_fp8_e8m0;
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
using e8m0_t = uint8_t;

// [128,4] rowwise and [4,128] colwise alignment requirements for the tensor with scaling factors
constexpr size_t scale_tensor_alignment_X_rowwise = 4;
constexpr size_t scale_tensor_alignment_Y_rowwise = 128;
constexpr size_t scale_tensor_alignment_X_colwise = 128;
constexpr size_t scale_tensor_alignment_Y_colwise = 4;

// Alignment requirements for the Tensor Memory Accelerator (TMA)
constexpr size_t TMA_GMEM_ALIGNMENT = 16;    // global memory address alignment
constexpr size_t TMA_SHMEM_ALIGNMENT = 128;  // shared memory address alignment

#define TRANSFORMER_ENGINE_SWITCH_CONDITION(CONDITION, FLAG, ...) \
  if (CONDITION) {                                                \
    constexpr bool FLAG = true;                                   \
    { __VA_ARGS__ }                                               \
  } else {                                                        \
    constexpr bool FLAG = false;                                  \
    { __VA_ARGS__ }                                               \
  }

inline bool is_aligned_ptr(const void *ptr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
  return (((x) + ((y)-1)) / (y));
}

template <typename T1, typename T2>
constexpr __device__ __host__ __forceinline__ uint64_t DIVUP_TO_MULTIPLE(const T1 &N, const T2 &M) {
  static_assert(std::is_integral<T1>::value && std::is_integral<T2>::value,
                "Integral type required.");
  return DIVUP(static_cast<uint64_t>(N), static_cast<uint64_t>(M)) * M;
}

// ###################################################

CUtensorMapDataType get_CUtensorMapDataType(at::ScalarType dtype) {
  static const std::unordered_map<at::ScalarType, CUtensorMapDataType> dtypeMapping = []() {
    std::unordered_map<at::ScalarType, CUtensorMapDataType> typeMapping = {
        {at::kByte, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
        {at::kFloat, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
        {at::kHalf, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
        {at::kBFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
        {at::Float8_e5m2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
        {at::Float8_e4m3fn, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8}};
    return typeMapping;
  }();
  return dtypeMapping.at(dtype);
}

// Set up parameters to create TMA descriptor.
void create_2D_tensor_map(CUtensorMap &tensorMap, const at::Tensor &tensor,
                          const uint64_t globalY, const uint64_t globalX, const uint32_t shmemY,
                          const uint32_t shmemX, const uint32_t stride_elems,
                          const uint32_t offset_elems, const size_t type_num_bits,
                          const CUtensorMapSwizzle swizzle) {
  // rank is the number of dimensions of the array
  constexpr uint32_t rank = 2;

  // Dimension for the packed data types must reflect the number of individual U# values.
  uint64_t size[rank] = {globalX, globalY};

  // The stride is the number of bytes to traverse from the first element of one row to the next
  uint64_t stride[rank - 1] = {(stride_elems * type_num_bits) / 8};

  // The boxSize is the size of the shared memory buffer that is used as the
  // source/destination of a TMA transfer
  uint32_t boxSize[rank] = {shmemX, shmemY};

  // The distance between elements in units of sizeof(element)
  uint32_t elemStride[rank] = {1, 1};

  const CUtensorMapDataType tensorDataType = get_CUtensorMapDataType(tensor.scalar_type());
  void *dataPtr = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(tensor.data()) +
                                           (offset_elems * type_num_bits) / 8);

  TORCH_CHECK(is_aligned_ptr(dataPtr, TMA_GMEM_ALIGNMENT),
             "Tensor data pointer must be 16B aligned");

  const int TMA_needed_size = (TMA_GMEM_ALIGNMENT * 8) / type_num_bits;
  TORCH_CHECK(globalX % TMA_needed_size == 0, "Shape not supported. For ", type_num_bits,
             "-bit data type, expected multiple of ", TMA_needed_size, ", got ", globalX);

  // Create the tensor descriptor.
  cuTensorMapEncodeTiled(
      &tensorMap,  // CUtensorMap *tensorMap,
      tensorDataType,
      rank,        // cuuint32_t tensorRank,
      dataPtr,     // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      boxSize,     // const cuuint32_t *boxDim,
      elemStride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

      // Swizzling can be used to avoid shared memory bank conflicts.
      swizzle,

      // L2 Promotion can be used to widen the effect of a cache-policy to a wider
      // set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,

      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}