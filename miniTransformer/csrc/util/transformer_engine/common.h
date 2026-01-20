#pragma once

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
using bf16 = __nv_bfloat16;
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

uint32_t _getAlignment(uintptr_t address);

// ###################################################

CUtensorMapDataType get_CUtensorMapDataType(at::ScalarType dtype);

// Set up parameters to create TMA descriptor.
void create_2D_tensor_map(CUtensorMap &tensorMap, const at::Tensor &tensor,
                          const uint64_t globalY, const uint64_t globalX, const uint32_t shmemY,
                          const uint32_t shmemX, const uint32_t stride_elems,
                          const uint32_t offset_elems, const size_t type_num_bits,
                          const CUtensorMapSwizzle swizzle);