/*************************************************************************
 * Shims for TE-aligned NVFP4 quantize_transpose_tuned_1D (ported header).
 * Provides transformer_engine::detail / dispatch::common / dispatch::nvfp4::core
 * and namespace ptx symbols expected by quantize_transpose_nvfp4_tuned_1d.cuh.
 ************************************************************************/
#pragma once

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

#include "common.h"
#include "curanddx.hpp"
#include "logging.h"
#include "ptx.cuh"
#include "vector.cuh"

namespace transformer_engine {
namespace detail {

template <typename T>
struct TypeExtrema {
  static constexpr float max = static_cast<float>(std::numeric_limits<T>::max());
};

#if FP4_TYPE_SUPPORTED
template <>
struct TypeExtrema<fp4e2m1> {
  static constexpr float max = 6.0f;
  static constexpr float max_inverse = 1.0f / max;
};
#endif

template <>
struct TypeExtrema<fp8e4m3> {
  static constexpr float max = 448.0f;
  static constexpr float max_inverse = 1.0f / max;
};

template <>
struct TypeExtrema<bf16> {
  static constexpr float max = 0x1.FEp127f;
};

}  // namespace detail

namespace dispatch {
namespace common {

__device__ __forceinline__ unsigned char *align_smem_ptr_per_TMA_requirements(unsigned char *p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  size_t addr = reinterpret_cast<size_t>(p);
  addr = (addr + TMA_SHMEM_ALIGNMENT - 1) & ~(TMA_SHMEM_ALIGNMENT - 1);
  return reinterpret_cast<unsigned char *>(addr);
#else
  return p;
#endif
}

}  // namespace common

namespace nvfp4 {

using nvfp4_scale_t = fp8e4m3;

namespace quantization_and_transposition_SF {
#if FP4_TYPE_SUPPORTED
__device__ __forceinline__ nvfp4_scale_t compute_decoding_scaling_factor(const float block_amax,
                                                                       const float S_enc) {
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;
  constexpr float fp4_max_inv = 1.0f / fp4_max;
  const float S_dec_b = block_amax * (S_enc * fp4_max_inv);
  return static_cast<nvfp4_scale_t>(fminf(S_dec_b, TypeExtrema<float>::max));
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace quantization_and_transposition_SF

namespace core {
#if FP4_TYPE_SUPPORTED
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  constexpr float fp8_max = TypeExtrema<fp8e4m3>::max;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

__device__ __forceinline__ uint32_t get_rbits(
    transformer_engine::curanddx::detail::philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS> &rng,
    uint4 &random_uint4, int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    random_uint4 = rng.generate4();
  }
  const uint32_t *const rbits_arr = reinterpret_cast<const uint32_t *>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace core

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

namespace ptx {

using ::mbarrier_init;
using ::mbarrier_invalid;
using ::mbarrier_arrive;
using ::mbarrier_arrive_expect_tx;
using ::mbarrier_wait_parity;
using ::fence_proxy_async_shared_cta;
using ::cp_async_bulk_tensor_2d_global_to_shared;
using ::cp_async_bulk_tensor_2d_shared_to_global;
using ::cp_async_bulk_commit_group;

template <typename T>
using FPx2 = vector::FPx2<T>;

__device__ __forceinline__ void mbarrier_wait_parity_acquire_cta_shared_cta(uint64_t *mbar,
                                                                              uint32_t phase_parity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
      "{\n\t"
      ".reg .b64 r1; \n\t"
      ".reg .pred waitComplete; \n\t"
      "WAIT: \n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  waitComplete, [%0], %1; \n\t"
      "@waitComplete bra DONE; \n\t"
      "bra WAIT; \n\t"
      "DONE: \n\t"
      "}\n\t"
      :
      : "r"(mbar_ptr), "r"(phase_parity)
      : "memory");
#else
  NVTE_DEVICE_ERROR("mbarrier_wait_parity_acquire_cta_shared_cta requires SM 10.0+.");
#endif
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cta_relaxed_shared_cta(uint64_t *mbar,
                                                                                 const uint32_t tx_count) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_ptr),
               "r"(tx_count));
#else
  NVTE_DEVICE_ERROR("mbarrier_arrive_expect_tx_cta_relaxed_shared_cta requires SM 10.0+.");
#endif
}

__device__ __forceinline__ void try_cancel_cta(uint64_t *mbar, __uint128_t *response_data_ptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
  asm volatile(
      "clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::"
      "all.b128 "
      "[%0], [%1];" ::"r"(workID_response),
      "r"(mbar_ptr));
#else
  NVTE_DEVICE_ERROR("try_cancel_cta requires SM 10.0+.");
#endif
}

__device__ __forceinline__ void get_cancelled_cta_id_2D(__uint128_t *response_data_ptr, int32_t &ctaid_X,
                                                        int32_t &ctaid_Y) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
  asm volatile(
      "{\n\t"
      ".reg .s32 x_ctaid; \n\t"
      ".reg .s32 y_ctaid; \n\t"
      "mov .s32 x_ctaid, -1; \n\t"
      "mov .s32 y_ctaid, -1; \n\t"
      ".reg.b128 try_cancel_response; \n\t"
      "ld.shared.b128 try_cancel_response, [%2]; \n\t"
      ".reg .pred P1; \n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P1, try_cancel_response; \n\t"
      "@P1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x_ctaid, y_ctaid, _, "
      "_}, try_cancel_response; \n\t"
      "mov .s32 %0, x_ctaid; \n\t"
      "mov .s32 %1, y_ctaid; \n\t"
      "}\n\t"
      : "=r"(ctaid_X), "=r"(ctaid_Y)
      : "r"(workID_response)
      : "memory");
#else
  NVTE_DEVICE_ERROR("get_cancelled_cta_id_2D requires SM 10.0+.");
#endif
}

__device__ __forceinline__ vector::bf16x2 ld_shared_b32(const vector::bf16x2 *__restrict__ src_smem) {
  vector::bf16x2 dst{};
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t src_smem_ptr = __cvta_generic_to_shared(src_smem);
  asm volatile("ld.shared.b32 %0, [%1];"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(src_smem_ptr));
#endif
  return dst;
}

__device__ __forceinline__ __uint128_t ld_shared_b128(const bf16 *__restrict__ src_smem) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint64_t elts03, elts47;
  const uint32_t src_smem_ptr = __cvta_generic_to_shared(src_smem);
  asm volatile(
      "{\n\t"
      ".reg.b128 xy; \n\t"
      "ld.shared.b128 xy, [%2]; \n\t"
      "mov.b128 {%0, %1}, xy; \n"
      "}\n"
      : "=l"(elts03), "=l"(elts47)
      : "r"(src_smem_ptr));
  return (static_cast<__uint128_t>(elts47) << 64) | static_cast<__uint128_t>(elts03);
#else
  NVTE_DEVICE_ERROR("ld_shared_b128 requires SM 10.0+.");
  return 0;
#endif
}

#if FP4_TYPE_SUPPORTED
__device__ __forceinline__ void st_shared_b32(fp4e2m1x2 *__restrict__ dst_smem, uint32_t fp4_pack_x8) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t dst_smem_ptr = __cvta_generic_to_shared(dst_smem);
  asm volatile("st.shared.b32 [%0], %1;" : : "r"(dst_smem_ptr), "r"(fp4_pack_x8));
#else
  NVTE_DEVICE_ERROR("st_shared_b32 requires SM 10.0+.");
#endif
}

__device__ __forceinline__ void st_shared_b64(fp4e2m1x2 *__restrict__ dst_smem, uint64_t fp4_pack_x16) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t dst_smem_ptr = __cvta_generic_to_shared(dst_smem);
  asm volatile("st.shared.b64 [%0], %1;" : : "r"(dst_smem_ptr), "l"(fp4_pack_x16));
#else
  NVTE_DEVICE_ERROR("st_shared_b64 requires SM 10.0+.");
#endif
}
#endif

__device__ __forceinline__ void abs_max_2x(vector::bf16x2 &dst, const vector::bf16x2 &p1,
                                         const vector::bf16x2 &p2) {
  ::abs_max_2x(dst, p1, p2);
}

template <size_t W>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (W == 0) {
    asm volatile("cp.async.bulk.wait_group.read 0;");
  } else if constexpr (W == 1) {
    asm volatile("cp.async.bulk.wait_group.read 1;");
  } else if constexpr (W == 2) {
    asm volatile("cp.async.bulk.wait_group.read 2;");
  } else if constexpr (W == 4) {
    asm volatile("cp.async.bulk.wait_group.read 4;");
  } else {
    NVTE_DEVICE_ERROR("cp_async_bulk_wait_group_read: unsupported W.");
  }
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_wait_group_read is only supported on SM 9.0+.");
#endif
}

#if FP4_TYPE_SUPPORTED
template <typename SCALING_COEFFICIENT_TYPE>
__device__ __forceinline__ uint32_t mul_cvt_bf16_to_fp4_8x_round_to_nearest(
    const uint64_t in03, const uint64_t in47, const SCALING_COEFFICIENT_TYPE scaling_coefficient) {
  uint32_t out_8x = 0;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, bf16>::value) {
    asm volatile(
        "{\n"
        ".reg.f32 zero; \n\t"
        "mov.b32 zero, 0; \n\t"
        ".reg.b16 scaling_coeff; \n\t"
        "mov.b16 scaling_coeff, %3; \n\t"
        ".reg.b16 v0_h, v1_h, v2_h, v3_h, v4_h, v5_h, v6_h, v7_h; \n\t"
        "mov.b64 {v0_h, v1_h, v2_h, v3_h}, %1; \n\t"
        "mov.b64 {v4_h, v5_h, v6_h, v7_h}, %2; \n\t"

        ".reg.f32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
        "fma.rn.f32.bf16 v0, v0_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v1, v1_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v2, v2_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v3, v3_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v4, v4_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v5, v5_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v6, v6_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v7, v7_h, scaling_coeff, zero; \n\t"

        ".reg.b8 f0, f1, f2, f3; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v1, v0;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v3, v2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f2, v5, v4;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f3, v7, v6;\n\t"
        "mov.b32 %0, {f0, f1, f2, f3};\n"
        "}"
        : "=r"(out_8x)
        : "l"(in03), "l"(in47), "h"(reinterpret_cast<const uint16_t &>(scaling_coefficient)));
  } else if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, float>::value) {
    asm volatile(
        "{\n"
        ".reg.b64 scaling_coeff_2x; \n\t"
        "mov.b64 scaling_coeff_2x, {%3, %3}; \n\t"
        ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
        "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"

        ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "cvt.f32.bf16 v4, v4_bf16; \n\t"
        "cvt.f32.bf16 v5, v5_bf16; \n\t"
        "cvt.f32.bf16 v6, v6_bf16; \n\t"
        "cvt.f32.bf16 v7, v7_bf16; \n\t"

        ".reg.b64 v01, v23, v45, v67; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mov.b64 v45, {v4, v5}; \n\t"
        "mov.b64 v67, {v6, v7}; \n\t"
        "mul.f32x2 v01, v01, scaling_coeff_2x; \n\t"
        "mul.f32x2 v23, v23, scaling_coeff_2x; \n\t"
        "mul.f32x2 v45, v45, scaling_coeff_2x; \n\t"
        "mul.f32x2 v67, v67, scaling_coeff_2x; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "mov.b64 {v5, v4}, v45; \n\t"
        "mov.b64 {v7, v6}, v67; \n\t"

        ".reg.b8 f0, f1, f2, f3; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f2, v4, v5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f3, v6, v7;\n\t"
        "mov.b32 %0, {f0, f1, f2, f3};\n\t"
        "}"
        : "=r"(out_8x)
        : "l"(in03), "l"(in47), "f"(scaling_coefficient));
  } else {
    NVTE_DEVICE_ERROR("Not supported scaling coefficient type.");
  }
#else
  NVTE_DEVICE_ERROR("FP4 cvt PTX instructions require SM 10.0+.");
#endif
  return out_8x;
}

template <typename SCALING_COEFFICIENT_TYPE>
__device__ __forceinline__ uint32_t mul_cvt_bf16_to_fp4_8x_stochastic_rounding(
    const uint64_t in03, const uint64_t in47, const SCALING_COEFFICIENT_TYPE scaling_coefficient,
    const uint32_t rbits03, const uint32_t rbits47) {
  uint32_t out_8x = 0;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, bf16>::value) {
    asm volatile(
        "{\n"
        ".reg.f32 zero; \n\t"
        "mov.b32 zero, 0; \n\t"
        ".reg.b16 scaling_coeff; \n\t"
        "mov.b16 scaling_coeff, %3; \n\t"
        ".reg.b16 v0_h, v1_h, v2_h, v3_h, v4_h, v5_h, v6_h, v7_h; \n\t"
        "mov.b64 {v0_h, v1_h, v2_h, v3_h}, %1; \n\t"
        "mov.b64 {v4_h, v5_h, v6_h, v7_h}, %2; \n\t"

        ".reg.f32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
        "fma.rn.f32.bf16 v0, v0_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v1, v1_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v2, v2_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v3, v3_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v4, v4_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v5, v5_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v6, v6_h, scaling_coeff, zero; \n\t"
        "fma.rn.f32.bf16 v7, v7_h, scaling_coeff, zero; \n\t"

        ".reg.b16 b03, b47; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, %4; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, %5; \n\t"
        "mov.b32 %0, {b03, b47};\n"
        "}"
        : "=r"(out_8x)
        : "l"(in03), "l"(in47), "h"(reinterpret_cast<const uint16_t &>(scaling_coefficient)),
          "r"(rbits03), "r"(rbits47));
  } else if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, float>::value) {
    asm volatile(
        "{\n"
        ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
        "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"

        ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "cvt.f32.bf16 v4, v4_bf16; \n\t"
        "cvt.f32.bf16 v5, v5_bf16; \n\t"
        "cvt.f32.bf16 v6, v6_bf16; \n\t"
        "cvt.f32.bf16 v7, v7_bf16; \n\t"

        "mul.f32 v0, v0, %3; \n\t"
        "mul.f32 v1, v1, %3; \n\t"
        "mul.f32 v2, v2, %3; \n\t"
        "mul.f32 v3, v3, %3; \n\t"
        "mul.f32 v4, v4, %3; \n\t"
        "mul.f32 v5, v5, %3; \n\t"
        "mul.f32 v6, v6, %3; \n\t"
        "mul.f32 v7, v7, %3; \n\t"
        ".reg.b16 b03, b47; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, %4; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, %5; \n\t"
        "mov.b32 %0, {b03, b47};\n"
        "}"
        : "=r"(out_8x)
        : "l"(in03), "l"(in47), "f"(scaling_coefficient), "r"(rbits03), "r"(rbits47));
  } else {
    NVTE_DEVICE_ERROR("Not supported scaling coefficient type.");
  }
#else
  NVTE_DEVICE_ERROR("FP4 stochastic cvt requires SM 10.0+.");
#endif
  return out_8x;
}
#endif  // FP4_TYPE_SUPPORTED

}  // namespace ptx
