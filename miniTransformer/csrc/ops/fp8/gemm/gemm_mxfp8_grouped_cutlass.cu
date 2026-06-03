/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Zero-overhead grouped MXFP8 GEMM (SM100, 1SM / 2SM schedule).
 *
 * Based on CUTLASS example 92_blackwell_moe_gemm_blockscaled_rcgrouped,
 * re-engineered to match the zero-overhead pattern of gemm_grouped_cutlass_v2.cu:
 *   - No DeviceAllocation / cudaMalloc per call.
 *   - No synchronous cudaMemcpy per call.
 *   - A GPU micro-kernel fills all CUTLASS metadata on device.
 *   - Gemm object cached; initialize() runs once per (num_groups, device).
 *   - Supports 1SM (default, small batch) and 2SM (high throughput) via use_2sm flag.
 *
 * Expected tensor layout:
 *   A   : [total_M, K]     uint8 (MXFP8 E4M3, 1 byte/elem), row-major, contiguous
 *   B   : [total_N, K]     uint8 (MXFP8 E4M3, 1 byte/elem), row-major, contiguous
 *   A_sf: contiguous, swizzled SFA for entire A (float_ue8m0_t / E8M0)
 *   B_sf: contiguous, swizzled SFB for entire B
 *   D   : [total_M, N]     bf16, row-major, pre-allocated
 *
 * Offset tensors (device, int32/int64):
 *   a_row_offsets[g]  : row index into A where group g starts  (len = groups+1)
 *   b_row_offsets[g]  : row index into B where group g starts  (len = groups+1)
 *   a_sf_offsets[g]   : element index into A_sf for group g    (len = groups+1)
 *   b_sf_offsets[g]   : element index into B_sf for group g    (len = groups+1)
 *
 * Persistent device buffers (pre-allocated by caller, never freed between calls):
 *   buf_ptrs          : [6 * max_groups]  uint8 (6 data ptr arrays)
 *   buf_strides       : [max_groups * (4 * sizeof(StrideA))]  uint8
 *   buf_layouts       : [max_groups * (2 * sizeof(LayoutSFA))]  uint8
 *   buf_problem_sizes : [max_groups * 3]  int32  (device copy of per-group (M_g,N_g,K))
 *
 * problem_sizes_host  : CPU tensor [groups * 3]  int32
 *   CUTLASS's grouped TMA kernel requires host-side problem sizes during
 *   gemm.initialize() to configure TMA descriptors.
 */

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
#include "cutlass/numeric_types.h"
#include "cutlass/float8.h"

#include "../../../ops.h"

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

using namespace cute;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
namespace mxfp8 {

using ElementInput       = cutlass::float_e4m3_t;
using ElementA           = cutlass::mx_float8_t<ElementInput>;
using ElementB           = cutlass::mx_float8_t<ElementInput>;
using ElementSF          = cutlass::float_ue8m0_t;   // E8M0 input scale factors
using ElementC           = cutlass::bfloat16_t;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ArchTag            = cutlass::arch::Sm100;
using LayoutATag         = cutlass::layout::RowMajor *;
using LayoutBTag         = cutlass::layout::ColumnMajor *;
using LayoutCTag         = cutlass::layout::RowMajor *;
using LayoutDTag         = cutlass::layout::RowMajor *;

// MXFP8: 16 elements per 128-bit TMA word (8 bits/elem)
static constexpr int AlignmentA = 16;
static constexpr int AlignmentB = 16;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ClusterShape = Shape<int32_t, int32_t, _1>;

// 1SM config
struct MMA1SMConfig {
    using MmaTileShape     = Shape<_128, _256, _128>;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// 2SM config
struct MMA2SMConfig {
    using MmaTileShape     = Shape<_256, _256, _128>;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

// Helper to build Collective types for a given MMA config
template <typename MMAConfig>
struct GemmTypes {
    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            typename MMAConfig::MmaTileShape, ClusterShape,
            Shape<_128, _64>,
            ElementAccumulator, ElementAccumulator,
            ElementC, LayoutCTag, AlignmentC,
            ElementD, LayoutDTag, AlignmentD,
            typename MMAConfig::EpilogueSchedule
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            ElementA, LayoutATag, AlignmentA,
            ElementB, LayoutBTag, AlignmentB,
            ElementAccumulator,
            typename MMAConfig::MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            typename MMAConfig::KernelSchedule
        >::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<
            ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Gemm1SM = typename GemmTypes<MMA1SMConfig>::Gemm;
using Gemm2SM = typename GemmTypes<MMA2SMConfig>::Gemm;

// Stride / Layout types — same for both 1SM and 2SM (same A/B/C/D shapes)
using StrideA   = typename Gemm1SM::GemmKernel::InternalStrideA;
using StrideB   = typename Gemm1SM::GemmKernel::InternalStrideB;
using StrideC   = typename Gemm1SM::GemmKernel::InternalStrideC;
using StrideD   = typename Gemm1SM::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm1SM::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm1SM::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm1SM::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Raw element types for pointer arithmetic (strip mx_float8_t wrapper)
using ArrayElementA = typename GemmTypes<MMA1SMConfig>::CollectiveMainloop::ArrayElementA;
using ArrayElementB = typename GemmTypes<MMA1SMConfig>::CollectiveMainloop::ArrayElementB;

// ---------------------------------------------------------------------------
// GPU micro-kernel: fills all CUTLASS grouped-GEMM metadata on device.
// Launch: <<<1, num_groups, 0, stream>>>
// ---------------------------------------------------------------------------
__global__ void fill_mxfp8_grouped_gemm_args_kernel(
    // Pointer arrays (output)
    const ArrayElementA** ptr_A,
    const ArrayElementB** ptr_B,
    const ElementSF**     ptr_SFA,
    const ElementSF**     ptr_SFB,
    const ElementC**      ptr_C,
    ElementD**            ptr_D,
    // Stride / layout arrays (output)
    StrideA*   stride_A,
    StrideB*   stride_B,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    StrideC*   stride_C,
    StrideD*   stride_D,
    // Problem sizes (device copy)
    typename ProblemShape::UnderlyingProblemShape* problem_sizes_dev,
    // Base data pointers
    const ArrayElementA*  a_base,
    const ArrayElementB*  b_base,
    const ElementSF*      sfa_base,
    const ElementSF*      sfb_base,
    ElementD*             d_base,
    // Offset arrays (device)
    const int32_t*        a_row_offsets,   // [groups+1] rows into A / D
    const int32_t*        b_row_offsets,   // [groups+1] rows into B
    const int64_t*        a_sf_offsets,    // [groups+1] elements into A_sf
    const int64_t*        b_sf_offsets,    // [groups+1] elements into B_sf
    int K,   // FP8 elements (1 byte each)
    int N    // output columns (same for every group)
)
{
    const int g   = threadIdx.x;
    const int M_g = a_row_offsets[g + 1] - a_row_offsets[g];
    const int N_g = b_row_offsets[g + 1] - b_row_offsets[g];

    ptr_A[g]   = a_base   + (int64_t)a_row_offsets[g] * K;
    ptr_B[g]   = b_base   + (int64_t)b_row_offsets[g] * K;
    ptr_SFA[g] = sfa_base + a_sf_offsets[g];
    ptr_SFB[g] = sfb_base + b_sf_offsets[g];
    ptr_C[g]   = reinterpret_cast<const ElementC*>(d_base)
                   + (int64_t)a_row_offsets[g] * N;
    ptr_D[g]   = d_base + (int64_t)a_row_offsets[g] * N;

    stride_A[g] = cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1});
    stride_B[g] = cutlass::make_cute_packed_stride(StrideB{}, {N_g, K, 1});
    stride_C[g] = cutlass::make_cute_packed_stride(StrideC{}, {M_g, N_g, 1});
    stride_D[g] = cutlass::make_cute_packed_stride(StrideD{}, {M_g, N_g, 1});

    auto shape_g = cute::make_shape(M_g, N_g, K, 1);
    layout_SFA[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(shape_g);
    layout_SFB[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(shape_g);

    problem_sizes_dev[g]    = cute::make_shape(M_g, N_g, K);
}

} // namespace mxfp8

// ---------------------------------------------------------------------------
// Gemm object cache keyed by (num_groups, device_id).
// ---------------------------------------------------------------------------
struct MxFP8GemmCacheEntry {
    mxfp8::Gemm1SM gemm1sm;
    mxfp8::Gemm2SM gemm2sm;
    bool initialized_1sm = false;
    bool initialized_2sm = false;
};

static std::unordered_map<int64_t, MxFP8GemmCacheEntry> s_mxfp8_gemm_cache;

static inline int64_t mxfp8_cache_key(int num_groups, int device_id) {
    return (int64_t)device_id * 100000 + num_groups;
}

// ---------------------------------------------------------------------------
// Buffer size query
// ---------------------------------------------------------------------------
std::vector<int64_t> mxfp8_grouped_gemm_buffer_sizes(int max_groups)
{
    using namespace mxfp8;
    // ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_C, ptr_D  (6 ptr arrays)
    int64_t ptrs_bytes    = (int64_t)6 * max_groups * sizeof(void*);
    int64_t strides_bytes = (int64_t)max_groups *
                            (sizeof(StrideA) + sizeof(StrideB) +
                             sizeof(StrideC) + sizeof(StrideD));
    int64_t layouts_bytes = (int64_t)max_groups *
                            (sizeof(LayoutSFA) + sizeof(LayoutSFB));
    return {ptrs_bytes, strides_bytes, layouts_bytes};
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
void mxfp8_grouped_gemm(
    at::Tensor&        D,
    const at::Tensor&  A,
    const at::Tensor&  B,
    const at::Tensor&  A_sf,
    const at::Tensor&  B_sf,
    const at::Tensor&  a_row_offsets,
    const at::Tensor&  b_row_offsets,
    const at::Tensor&  a_sf_offsets,
    const at::Tensor&  b_sf_offsets,
    int                num_groups,
    at::Tensor&        buf_ptrs,
    at::Tensor&        buf_strides,
    at::Tensor&        buf_layouts,
    at::Tensor&        buf_problem_sizes_dev,
    const at::Tensor&  problem_sizes_host,
    at::Tensor&        workspace,
    bool               use_2sm)
{
    using namespace mxfp8;

    TORCH_CHECK(num_groups > 0);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index());

    const int K = static_cast<int>(A.size(1));  // FP8: 1 byte/elem, K = cols
    const int N = static_cast<int>(D.size(1));

    // ------------------------------------------------------------------
    // Sub-buffer pointers from packed persistent buffers
    // ------------------------------------------------------------------
    auto* raw_ptrs_u8 = reinterpret_cast<uint8_t*>(buf_ptrs.data_ptr());
    auto* raw_strides = reinterpret_cast<uint8_t*>(buf_strides.data_ptr());
    auto* raw_layouts = reinterpret_cast<uint8_t*>(buf_layouts.data_ptr());

    const size_t ptr_sz = sizeof(void*);
    auto* ptr_A   = reinterpret_cast<const ArrayElementA**>(raw_ptrs_u8 + 0 * num_groups * ptr_sz);
    auto* ptr_B   = reinterpret_cast<const ArrayElementB**>(raw_ptrs_u8 + 1 * num_groups * ptr_sz);
    auto* ptr_SFA = reinterpret_cast<const ElementSF**>    (raw_ptrs_u8 + 2 * num_groups * ptr_sz);
    auto* ptr_SFB = reinterpret_cast<const ElementSF**>    (raw_ptrs_u8 + 3 * num_groups * ptr_sz);
    auto* ptr_C   = reinterpret_cast<const ElementC**>     (raw_ptrs_u8 + 4 * num_groups * ptr_sz);
    auto* ptr_D   = reinterpret_cast<ElementD**>           (raw_ptrs_u8 + 5 * num_groups * ptr_sz);

    auto* stride_A   = reinterpret_cast<StrideA*>  (raw_strides);
    auto* stride_B   = reinterpret_cast<StrideB*>  (stride_A + num_groups);
    auto* stride_C   = reinterpret_cast<StrideC*>  (stride_B + num_groups);
    auto* stride_D   = reinterpret_cast<StrideD*>  (stride_C + num_groups);

    auto* layout_SFA = reinterpret_cast<LayoutSFA*>(raw_layouts);
    auto* layout_SFB = reinterpret_cast<LayoutSFB*>(layout_SFA + num_groups);

    // problem_sizes_dev: [groups*3] int32 — one (M_g, N_g, K) per group
    auto* prob_dev  = reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
                          buf_problem_sizes_dev.data_ptr());
    auto* prob_host = reinterpret_cast<const typename ProblemShape::UnderlyingProblemShape*>(
                          problem_sizes_host.data_ptr());

    // ------------------------------------------------------------------
    // Step 1: fill all CUTLASS args on device
    // ------------------------------------------------------------------
    fill_mxfp8_grouped_gemm_args_kernel<<<1, num_groups, 0, stream>>>(
        ptr_A, ptr_B, ptr_SFA, ptr_SFB, ptr_C, ptr_D,
        stride_A, stride_B, layout_SFA, layout_SFB, stride_C, stride_D,
        prob_dev,
        static_cast<const ArrayElementA*>(A.data_ptr()),
        static_cast<const ArrayElementB*>(B.data_ptr()),
        static_cast<const ElementSF*>(A_sf.data_ptr()),
        static_cast<const ElementSF*>(B_sf.data_ptr()),
        static_cast<ElementD*>(D.data_ptr()),
        static_cast<const int32_t*>(a_row_offsets.data_ptr()),
        static_cast<const int32_t*>(b_row_offsets.data_ptr()),
        static_cast<const int64_t*>(a_sf_offsets.data_ptr()),
        static_cast<const int64_t*>(b_sf_offsets.data_ptr()),
        K, N
    );

    // ------------------------------------------------------------------
    // Step 2: build CUTLASS Arguments
    // ------------------------------------------------------------------
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id              = A.device().index();
    hw_info.sm_count               = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    hw_info.cluster_shape          = use_2sm ? dim3(2, 1, 1) : dim3(1, 1, 1);
    hw_info.cluster_shape_fallback = use_2sm ? dim3(2, 1, 1) : dim3(1, 1, 1);

    // GroupProblemShape args: {num_groups, prob_dev (device), prob_host (host)}
    // prob_dev[g] = {M_g, N_g, K} per group, filled by GPU micro-kernel above.
    int64_t key = mxfp8_cache_key(num_groups, A.device().index());
    auto& entry = s_mxfp8_gemm_cache[key];

    if (use_2sm) {
        typename Gemm2SM::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {num_groups, prob_dev, prob_host},
            {
                ptr_A,    stride_A,
                ptr_B,    stride_B,
                ptr_SFA,  layout_SFA,
                ptr_SFB,  layout_SFB
            },
            {
                {},
                ptr_C,  stride_C,
                ptr_D,  stride_D
            },
            hw_info
        };
        arguments.epilogue.thread.alpha = ElementAccumulator(1);
        arguments.epilogue.thread.beta  = ElementAccumulator(0);

        if (!entry.initialized_2sm) {
            CUTLASS_CHECK(entry.gemm2sm.can_implement(arguments));
            CUTLASS_CHECK(entry.gemm2sm.initialize(arguments, workspace.data_ptr(), stream));
            entry.initialized_2sm = true;
        }
        CUTLASS_CHECK(entry.gemm2sm.run(arguments, workspace.data_ptr(), stream));
    } else {
        typename Gemm1SM::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {num_groups, prob_dev, prob_host},
            {
                ptr_A,    stride_A,
                ptr_B,    stride_B,
                ptr_SFA,  layout_SFA,
                ptr_SFB,  layout_SFB
            },
            {
                {},
                ptr_C,  stride_C,
                ptr_D,  stride_D
            },
            hw_info
        };
        arguments.epilogue.thread.alpha = ElementAccumulator(1);
        arguments.epilogue.thread.beta  = ElementAccumulator(0);

        if (!entry.initialized_1sm) {
            CUTLASS_CHECK(entry.gemm1sm.can_implement(arguments));
            CUTLASS_CHECK(entry.gemm1sm.initialize(arguments, workspace.data_ptr(), stream));
            entry.initialized_1sm = true;
        }
        CUTLASS_CHECK(entry.gemm1sm.run(arguments, workspace.data_ptr(), stream));
    }
}
