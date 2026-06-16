/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Zero-overhead grouped FP4 GEMM with fused per-token scale (SM100, 2SM).
 *
 * Variant of gemm_grouped_cutlass_v2.cu where the post-GEMM
 *     D[m, n] *= per_token_scale[m]
 * step is folded into the CUTLASS epilogue via EVT (Sm90ColBroadcast over the
 * PtrArray epilogue).  This removes the separate per_token_scale_kernel_v2
 * launch and replaces a BF16-domain multiply with an FP32-domain multiply
 * fused into the accumulator → BF16 conversion (higher SNR, no extra HBM
 * round-trip on D).
 *
 * Public surface:
 *   grouped_cutlass_gemm_v2_evt_buffer_sizes(max_groups)
 *   grouped_cutlass_gemm_v2_evt(D, A, B, A_sf, B_sf, ..., per_token_scale)
 *
 * Persistent device buffers (caller-allocated, reused across calls):
 *   buf_ptrs          : [7 * max_groups]  int64  — A,B,SFA,SFB,C,D,Scale
 *   buf_strides       : 4 strides per group
 *   buf_layouts       : 2 SF layouts per group
 *   buf_problem_sizes : groups * 3  int32  (device copy)
 *
 * per_token_scale: optional, dtype float, shape [total_M].  When undefined or
 * empty, the epilogue ColBroadcast EnableNullptr path falls back to a unit
 * scale (i.e. plain D = bf16(acc)), so this entry point can also serve the
 * no-scale case without an extra branch on the caller side.
 *
 * Mathematical difference vs the unfused v2 path:
 *   v2:      D = bf16(acc);              D *= bf16(scale)   (two bf16 muls)
 *   v2_evt:  D = bf16( acc_fp32 * scale_fp32 )              (one fp32 mul)
 * The fused path is strictly higher-precision (FP32 multiply before the
 * acc→bf16 cast).  Bit-identical equivalence with v2 is therefore not
 * expected — SNR should improve slightly.
 */

#ifndef MINI_TE_NO_BLACKWELL_OPS

#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

// EVT fusion: per-token (column-broadcast) scale × accumulator → D.
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"

#include "../../../ops.h"

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

using namespace cute;

// ---------------------------------------------------------------------------
// Type aliases — distinct namespace so that CUTLASS instantiates a separate
// Gemm type and the v2 (unfused) types remain unaffected.
// ---------------------------------------------------------------------------
namespace v2_evt {

using ElementA           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD           = cutlass::bfloat16_t;
using ElementC           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ArchTag            = cutlass::arch::Sm100;
using LayoutATag         = cutlass::layout::RowMajor *;
using LayoutBTag         = cutlass::layout::ColumnMajor *;
using LayoutCTag         = cutlass::layout::RowMajor *;
using LayoutDTag         = cutlass::layout::RowMajor *;
using ElementSF          = cutlass::float_ue4m3_t;

static constexpr int AlignmentA = 32;
static constexpr int AlignmentB = 32;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ProblemShape     = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using MmaTileShape     = Shape<_256, _256, _256>;
using ClusterShape     = Shape<int32_t, int32_t, _1>;
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

// Per-token scale fusion:  D = bf16(  acc_fp32  *  scale_fp32[m]  ).
// Sm90ColBroadcast<float*, ...> auto-detects PtrArray (IsArrayOfPointers=true)
// and dereferences params.ptr_col[batch_idx] for per-batch column buffers.
// Blackwell (SM100) epilogue inherits Sm90 EVT nodes via FusionCallbacks
// (sm100_callbacks_tma_warpspecialized.hpp:130-149; see CUTLASS example 71).
using FusionOp = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, ElementAccumulator,
        cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90AccFetch,
    cutlass::epilogue::fusion::Sm90ColBroadcast<
        /*Stages=*/0,
        /*CtaTileShapeMNK=*/MmaTileShape,
        /*ElementInput_=*/ElementAccumulator*,  // float* → PtrArray semantics
        /*ElementCompute=*/ElementAccumulator,
        /*StrideMNL=*/cute::Stride<cute::_1, cute::_0, cute::_0>>>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        EpilogueSchedule,
        FusionOp
    >::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule
    >::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
using StrideC   = typename Gemm::GemmKernel::InternalStrideC;
using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ArrayElementA = typename CollectiveMainloop::ArrayElementA;
using ArrayElementB = typename CollectiveMainloop::ArrayElementB;

// ---------------------------------------------------------------------------
// GPU micro-kernel: fills all CUTLASS grouped-GEMM metadata on device.
// Identical to v2's version + one extra output: per-group ptr_Scale.
// Launch: <<<1, num_groups, 0, stream>>>
// ---------------------------------------------------------------------------
__global__ void fill_grouped_gemm_args_kernel(
    // Pointer arrays (output)
    const ArrayElementA**     ptr_A,
    const ArrayElementB**     ptr_B,
    const ElementSF**         ptr_SFA,
    const ElementSF**         ptr_SFB,
    const ElementC**          ptr_C,
    ElementD**                ptr_D,
    const ElementAccumulator** ptr_Scale,
    // Stride / layout arrays (output)
    StrideA*   stride_A,
    StrideB*   stride_B,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    StrideC*   stride_C,
    StrideD*   stride_D,
    // Problem sizes (output, device copy)
    typename ProblemShape::UnderlyingProblemShape* problem_sizes_dev,
    // Base data pointers
    const ArrayElementA*      a_base,
    const ArrayElementB*      b_base,
    const ElementSF*          sfa_base,
    const ElementSF*          sfb_base,
    ElementD*                 d_base,
    const ElementAccumulator* scale_base,    // [total_M] float, may be nullptr
    // Offset arrays (device)
    const int32_t* a_row_offsets,  // [groups+1]  rows into A / D / scale
    const int32_t* b_row_offsets,  // [groups+1]  rows into B
    const int64_t* a_sf_offsets,   // [groups+1]  elements into A_sf
    const int64_t* b_sf_offsets,   // [groups+1]  elements into B_sf
    int K_packed,   // A.size(1) == K // 2
    int K,          // actual K (FP4 elements)
    int N           // output columns (same for every group)
)
{
    const int g  = threadIdx.x;
    const int M_g = a_row_offsets[g + 1] - a_row_offsets[g];
    const int N_g = b_row_offsets[g + 1] - b_row_offsets[g];

    ptr_A[g]   = a_base   + (int64_t)a_row_offsets[g] * K_packed;
    ptr_B[g]   = b_base   + (int64_t)b_row_offsets[g] * K_packed;
    ptr_SFA[g] = sfa_base + a_sf_offsets[g];
    ptr_SFB[g] = sfb_base + b_sf_offsets[g];
    ptr_C[g]   = reinterpret_cast<const ElementC*>(d_base)
                   + (int64_t)a_row_offsets[g] * N;
    ptr_D[g]   = d_base + (int64_t)a_row_offsets[g] * N;

    // Per-group per-token scale pointer: scale_base is flat [total_M] float;
    // each group's first row index = a_row_offsets[g].  When scale_base is
    // nullptr the per-group entry is also nullptr, and the epilogue's
    // ColBroadcast EnableNullptr path falls back to null_default (set to 1
    // on the host side → no-op multiply).
    ptr_Scale[g] = (scale_base == nullptr)
        ? nullptr
        : scale_base + a_row_offsets[g];

    stride_A[g] = cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1});
    stride_B[g] = cutlass::make_cute_packed_stride(StrideB{}, {N_g, K, 1});
    stride_C[g] = cutlass::make_cute_packed_stride(StrideC{}, {M_g, N_g, 1});
    stride_D[g] = cutlass::make_cute_packed_stride(StrideD{}, {M_g, N_g, 1});

    auto shape_g = cute::make_shape(M_g, N_g, K, 1);
    layout_SFA[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(shape_g);
    layout_SFB[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(shape_g);

    problem_sizes_dev[g] = cute::make_shape(M_g, N_g, K);
}

} // namespace v2_evt

// ---------------------------------------------------------------------------
// Persistent Gemm instance cache keyed by (num_groups, device_id).
// Independent from the v2 cache: the EVT path is a distinct CUTLASS Gemm type.
// ---------------------------------------------------------------------------
struct GemmEvtCacheEntry {
    v2_evt::Gemm gemm;
    bool initialized = false;
};

static std::unordered_map<int64_t, GemmEvtCacheEntry> s_gemm_evt_cache;

static inline int64_t gemm_evt_cache_key(int num_groups, int device_id) {
    return (int64_t)device_id * 100000 + num_groups;
}

// ---------------------------------------------------------------------------
// Helper: return buffer sizes required by grouped_cutlass_gemm_v2_evt.
// 7 pointers/group (A,B,SFA,SFB,C,D,Scale) vs 6 in the unfused v2 path.
// ---------------------------------------------------------------------------
std::vector<int64_t> grouped_cutlass_gemm_v2_evt_buffer_sizes(int max_groups)
{
    using namespace v2_evt;
    int64_t ptrs_bytes    = (int64_t)7 * max_groups * sizeof(void*);
    int64_t strides_bytes = (int64_t)max_groups *
                            (sizeof(StrideA) + sizeof(StrideB) +
                             sizeof(StrideC) + sizeof(StrideD));
    int64_t layouts_bytes = (int64_t)max_groups *
                            (sizeof(LayoutSFA) + sizeof(LayoutSFB));
    return {ptrs_bytes, strides_bytes, layouts_bytes};
}

// ---------------------------------------------------------------------------
// Public entry point: grouped FP4 GEMM with EVT-fused per-token scale.
//
// buf_ptrs    [7*groups]  int64  device  — ptr_A,B,SFA,SFB,C,D,Scale
// buf_strides, buf_layouts, buf_problem_sizes_dev: same as v2
// per_token_scale: optional [total_M] float; empty → no-op (unit) scale
// ---------------------------------------------------------------------------
void grouped_cutlass_gemm_v2_evt(
    at::Tensor&       D,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& A_sf,
    const at::Tensor& B_sf,
    const at::Tensor& a_row_offsets,        // device int32 [groups+1]
    const at::Tensor& b_row_offsets,        // device int32 [groups+1]
    const at::Tensor& a_sf_offsets,         // device int64 [groups+1]
    const at::Tensor& b_sf_offsets,         // device int64 [groups+1]
    int               num_groups,
    at::Tensor&       buf_ptrs,
    at::Tensor&       buf_strides,
    at::Tensor&       buf_layouts,
    at::Tensor&       buf_problem_sizes_dev,
    const at::Tensor& problem_sizes_host,   // CPU int32 [groups*3]
    at::Tensor&       workspace,
    const at::Tensor& per_token_scale)
{
    using namespace v2_evt;

    TORCH_CHECK(num_groups > 0);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index());

    const int K_packed = static_cast<int>(A.size(1));
    const int K        = K_packed * 2;
    const int N        = static_cast<int>(D.size(1));

    // ------------------------------------------------------------------
    // Derive sub-buffer pointers from the packed persistent buffers
    // ------------------------------------------------------------------
    auto* raw_ptrs_u8 = reinterpret_cast<uint8_t*>(buf_ptrs.data_ptr());
    auto* raw_strides = reinterpret_cast<uint8_t*>(buf_strides.data_ptr());
    auto* raw_layouts = reinterpret_cast<uint8_t*>(buf_layouts.data_ptr());

    const size_t ptr_sz = sizeof(void*);
    auto* ptr_A     = reinterpret_cast<const ArrayElementA**>     (raw_ptrs_u8 + 0 * num_groups * ptr_sz);
    auto* ptr_B     = reinterpret_cast<const ArrayElementB**>     (raw_ptrs_u8 + 1 * num_groups * ptr_sz);
    auto* ptr_SFA   = reinterpret_cast<const ElementSF**>         (raw_ptrs_u8 + 2 * num_groups * ptr_sz);
    auto* ptr_SFB   = reinterpret_cast<const ElementSF**>         (raw_ptrs_u8 + 3 * num_groups * ptr_sz);
    auto* ptr_C     = reinterpret_cast<const ElementC**>          (raw_ptrs_u8 + 4 * num_groups * ptr_sz);
    auto* ptr_D     = reinterpret_cast<ElementD**>                (raw_ptrs_u8 + 5 * num_groups * ptr_sz);
    auto* ptr_Scale = reinterpret_cast<const ElementAccumulator**>(raw_ptrs_u8 + 6 * num_groups * ptr_sz);

    auto* stride_A   = reinterpret_cast<StrideA*>  (raw_strides);
    auto* stride_B   = reinterpret_cast<StrideB*>  (stride_A + num_groups);
    auto* stride_C   = reinterpret_cast<StrideC*>  (stride_B + num_groups);
    auto* stride_D   = reinterpret_cast<StrideD*>  (stride_C + num_groups);

    auto* layout_SFA = reinterpret_cast<LayoutSFA*>(raw_layouts);
    auto* layout_SFB = reinterpret_cast<LayoutSFB*>(layout_SFA + num_groups);

    auto* prob_dev   = reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
                           buf_problem_sizes_dev.data_ptr());
    auto* prob_host  = reinterpret_cast<const typename ProblemShape::UnderlyingProblemShape*>(
                           problem_sizes_host.data_ptr());

    const bool has_scale = per_token_scale.defined() && per_token_scale.numel() > 0;
    const ElementAccumulator* scale_base = has_scale
        ? static_cast<const ElementAccumulator*>(per_token_scale.data_ptr())
        : nullptr;

    // ------------------------------------------------------------------
    // Step 1: fill all CUTLASS args on device — zero host-device copies
    // ------------------------------------------------------------------
    v2_evt::fill_grouped_gemm_args_kernel<<<1, num_groups, 0, stream>>>(
        ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_C, ptr_D, ptr_Scale,
        stride_A, stride_B, layout_SFA, layout_SFB, stride_C, stride_D,
        prob_dev,
        static_cast<const ArrayElementA*>(A.data_ptr()),
        static_cast<const ArrayElementB*>(B.data_ptr()),
        static_cast<const ElementSF*>(A_sf.data_ptr()),
        static_cast<const ElementSF*>(B_sf.data_ptr()),
        static_cast<ElementD*>(D.data_ptr()),
        scale_base,
        static_cast<const int32_t*>(a_row_offsets.data_ptr()),
        static_cast<const int32_t*>(b_row_offsets.data_ptr()),
        static_cast<const int64_t*>(a_sf_offsets.data_ptr()),
        static_cast<const int64_t*>(b_sf_offsets.data_ptr()),
        K_packed, K, N
    );

    // ------------------------------------------------------------------
    // Step 2: build CUTLASS Arguments
    // ------------------------------------------------------------------
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id              = A.device().index();
    hw_info.sm_count               = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    hw_info.cluster_shape          = dim3(2, 1, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);

    // FusionOp::Arguments shape (tuple order = ChildOps..., NodeOp):
    //   { Sm90AccFetch_args, Sm90ColBroadcast_args, Sm90Compute_args }
    // ColBroadcast with float* ElementInput is PtrArray: ptr_col = device array
    // of per-batch float* pointers (one per group, written by fill kernel).
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups, prob_dev, prob_host},
        {
            ptr_A,    stride_A,
            ptr_B,    stride_B,
            ptr_SFA,  layout_SFA,
            ptr_SFB,  layout_SFB
        },
        {
            {  // FusionOp::Arguments
                {},  // Sm90AccFetch (empty)
                {    // Sm90ColBroadcast
                    ptr_Scale,                  // float* const* — per-batch
                    ElementAccumulator(1),      // null_default → unit scale
                    {}                          // dCol (default StrideMNL)
                },
                {}   // Sm90Compute<multiplies> (empty)
            },
            ptr_C,  stride_C,
            ptr_D,  stride_D
        },
        hw_info
    };

    // ------------------------------------------------------------------
    // Step 3: initialize (first call only) + run
    // ------------------------------------------------------------------
    int64_t key = gemm_evt_cache_key(num_groups, A.device().index());
    auto& entry = s_gemm_evt_cache[key];
    if (!entry.initialized) {
        CUTLASS_CHECK(entry.gemm.can_implement(arguments));
        CUTLASS_CHECK(entry.gemm.initialize(arguments, workspace.data_ptr(), stream));
        entry.initialized = true;
    }
    CUTLASS_CHECK(entry.gemm.run(arguments, workspace.data_ptr(), stream));
}

#else  // MINI_TE_NO_BLACKWELL_OPS — stubs for non-Blackwell builds

#include "../../../ops.h"
#include <stdexcept>

std::vector<int64_t> grouped_cutlass_gemm_v2_evt_buffer_sizes(int max_groups) {
  throw std::runtime_error("grouped_cutlass_gemm_v2_evt_buffer_sizes requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
  return {};
}

void grouped_cutlass_gemm_v2_evt(
    at::Tensor&       D,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& A_sf,
    const at::Tensor& B_sf,
    const at::Tensor& a_row_offsets,
    const at::Tensor& b_row_offsets,
    const at::Tensor& a_sf_offsets,
    const at::Tensor& b_sf_offsets,
    int               num_groups,
    at::Tensor&       buf_ptrs,
    at::Tensor&       buf_strides,
    at::Tensor&       buf_layouts,
    at::Tensor&       buf_problem_sizes_dev,
    const at::Tensor& problem_sizes_host,
    at::Tensor&       workspace,
    const at::Tensor& per_token_scale
) {
  throw std::runtime_error("grouped_cutlass_gemm_v2_evt requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

#endif  // MINI_TE_NO_BLACKWELL_OPS
