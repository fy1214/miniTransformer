#include "common.h"

uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (address % alignment == 0) {
      return alignment;
    }
  }
}

// ###################################################

CUtensorMapDataType get_CUtensorMapDataType(at::ScalarType dtype) {
  static const std::unordered_map<at::ScalarType, CUtensorMapDataType> dtypeMapping = []() {
    std::unordered_map<at::ScalarType, CUtensorMapDataType> typeMapping = {
        {at::ScalarType::Byte, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
        {at::ScalarType::Float, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
        {at::ScalarType::Half, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
        {at::ScalarType::BFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
        {at::ScalarType::Float8_e5m2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
        {at::ScalarType::Float8_e4m3fn, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8}};
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

  CUtensorMapDataType tensorDataType = get_CUtensorMapDataType(tensor.scalar_type());
#if FP4_TYPE_SUPPORTED
  // FP4 outputs/scales are stored in uint8 buffers but represent packed 4-bit data.
  // TMA descriptor must use U4 packed mode to match TE behavior.
  if (type_num_bits == 4) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
  }
#endif
  void *dataPtr = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(tensor.data_ptr()) +
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