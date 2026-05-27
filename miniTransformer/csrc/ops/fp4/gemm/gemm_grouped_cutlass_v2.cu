/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * Zero-overhead grouped FP4 GEMM (SM100, 2SM schedule).
 *
 * Differences vs gemm_grouped_cutlass.cu:
 *   - No DeviceAllocation / cudaMalloc per call.
 *   - No synchronous cudaMemcpy per call.
 *   - A GPU micro-kernel (fill_grouped_gemm_args_kernel) fills all CUTLASS
 *     metadata (ptr arrays, stride arrays, layout arrays, problem sizes) on
 *     device from contiguous A/B tensors and pre-computed offset arrays.
 *   - Caller pre-allocates persistent device buffers once; only workspace and
 *     ptr/stride/layout buffers need to live across calls.
 *
 * Expected tensor layout:
 *   A  : [total_M, K//2]  uint8, row-major, contiguous
 *   B  : [total_N, K//2]  uint8, row-major, contiguous
 *   A_sf: contiguous, swizzled SFA for the entire A
 *   B_sf: contiguous, swizzled SFB for the entire B
 *   D  : [total_M, N]     bf16, row-major, pre-allocated
 *
 * Offset tensors (device, int32/int64):
 *   a_row_offsets[g]  : row index into A where group g starts  (len = groups+1)
 *   b_row_offsets[g]  : row index into B where group g starts  (len = groups+1)
 *   a_sf_offsets[g]   : element index into A_sf for group g    (len = groups+1)
 *   b_sf_offsets[g]   : element index into B_sf for group g    (len = groups+1)
 *
 * Persistent device buffers (pre-allocated by caller, never freed between calls):
 *   buf_ptrs          : [6 * max_groups]  int64
 *   buf_strides       : [max_groups * (4 * sizeof(StrideA))]  uint8
 *   buf_layouts       : [max_groups * (2 * sizeof(LayoutSFA))]  uint8
 *   buf_problem_sizes : [max_groups * 3]  int32  (device copy)
 *
 * problem_sizes_host  : CPU tensor [groups * 3]  int32
 *   CUTLASS's grouped TMA kernel requires host-side problem sizes during
 *   gemm.initialize() to configure TMA descriptors.  The caller must supply
 *   this pre-filled tensor (trivial to compute: M_g = a_row_offsets[g+1] - [g],
 *   N_g = b_row_offsets[g+1] - [g], K = fixed).
 */

#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "../../../ops.h"

// Per-token scale: D[m][n] *= scale[m], one CTA per row.
template <typename T>
__global__ void per_token_scale_kernel_v2(T* __restrict__ D, const float* __restrict__ scale, int N)
{
    const int m = blockIdx.x;
    const float s = scale[m];
    T* row = D + (int64_t)m * N;
    for (int n = threadIdx.x; n < N; n += blockDim.x)
        row[n] = static_cast<T>(static_cast<float>(row[n]) * s);
}

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

using namespace cute;

// ---------------------------------------------------------------------------
// Type aliases (identical to gemm_grouped_cutlass.cu)
// ---------------------------------------------------------------------------
namespace v2 {

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

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        EpilogueSchedule
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
// Launch: <<<1, num_groups, 0, stream>>>
// ---------------------------------------------------------------------------
__global__ void fill_grouped_gemm_args_kernel(
    // Pointer arrays (output)
    const ArrayElementA** ptr_A,
    const ArrayElementB** ptr_B,
    const ElementSF**     ptr_SFA,
    const ElementSF**     ptr_SFB,
    const ElementC**      ptr_C,
    ElementD**            ptr_D,
    // Stride / layout arrays (output)
    StrideA*  stride_A,
    StrideB*  stride_B,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB,
    StrideC*  stride_C,
    StrideD*  stride_D,
    // Problem sizes (output, device copy)
    typename ProblemShape::UnderlyingProblemShape* problem_sizes_dev,
    // Base data pointers
    const ArrayElementA* a_base,
    const ArrayElementB* b_base,
    const ElementSF*     sfa_base,
    const ElementSF*     sfb_base,
    ElementD*            d_base,
    // Offset arrays (device)
    const int32_t* a_row_offsets,  // [groups+1]  rows into A / D
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

    // Pointers into contiguous A / B / D
    ptr_A[g]   = a_base   + (int64_t)a_row_offsets[g] * K_packed;
    ptr_B[g]   = b_base   + (int64_t)b_row_offsets[g] * K_packed;
    ptr_SFA[g] = sfa_base + a_sf_offsets[g];
    ptr_SFB[g] = sfb_base + b_sf_offsets[g];
    ptr_C[g]   = reinterpret_cast<const ElementC*>(d_base)
                   + (int64_t)a_row_offsets[g] * N;
    ptr_D[g]   = d_base + (int64_t)a_row_offsets[g] * N;

    // Strides
    stride_A[g] = cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1});
    stride_B[g] = cutlass::make_cute_packed_stride(StrideB{}, {N_g, K, 1});
    stride_C[g] = cutlass::make_cute_packed_stride(StrideC{}, {M_g, N_g, 1});
    stride_D[g] = cutlass::make_cute_packed_stride(StrideD{}, {M_g, N_g, 1});

    // Scale-factor layouts
    auto shape_g = cute::make_shape(M_g, N_g, K, 1);
    layout_SFA[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(shape_g);
    layout_SFB[g] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(shape_g);

    // Problem sizes (device copy used by the GEMM kernel itself)
    problem_sizes_dev[g] = cute::make_shape(M_g, N_g, K);
}

} // namespace v2

// ---------------------------------------------------------------------------
// Helper: return buffer sizes required by grouped_cutlass_gemm_v2.
// Call once to allocate persistent tensors on the Python side.
// Returns a dict: { "ptrs": int, "strides": int, "layouts": int }  (bytes)
// ---------------------------------------------------------------------------
std::vector<int64_t> grouped_cutlass_gemm_v2_buffer_sizes(int max_groups)
{
    using namespace v2;
    int64_t ptrs_bytes    = (int64_t)6 * max_groups * sizeof(void*);
    int64_t strides_bytes = (int64_t)max_groups *
                            (sizeof(StrideA) + sizeof(StrideB) +
                             sizeof(StrideC) + sizeof(StrideD));
    int64_t layouts_bytes = (int64_t)max_groups *
                            (sizeof(LayoutSFA) + sizeof(LayoutSFB));
    // problem_sizes: max_groups * 3 * sizeof(int32) — caller allocates directly
    return {ptrs_bytes, strides_bytes, layouts_bytes};
}

// ---------------------------------------------------------------------------
// Public entry point: zero-overhead grouped FP4 GEMM.
//
// buf_ptrs    [6*groups]  int64  device  — ptr_A,B,SFA,SFB,C,D
// buf_strides [groups*(sizeof(StrideA)+sizeof(StrideB)+sizeof(StrideC)+sizeof(StrideD))] uint8 device
// buf_layouts [groups*(sizeof(LayoutSFA)+sizeof(LayoutSFB))] uint8 device
// buf_problem_sizes_dev [groups*3] int32 device
// problem_sizes_host    [groups*3] int32 CPU  — required by CUTLASS TMA init
// workspace             uint8 device — pre-allocated, reused across calls
// ---------------------------------------------------------------------------
void grouped_cutlass_gemm_v2(
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
    // Persistent device buffers (pre-allocated once)
    at::Tensor&       buf_ptrs,
    at::Tensor&       buf_strides,
    at::Tensor&       buf_layouts,
    at::Tensor&       buf_problem_sizes_dev,
    // Host problem sizes (required by CUTLASS TMA initialize)
    const at::Tensor& problem_sizes_host,   // CPU int32 [groups*3]
    at::Tensor&       workspace,
    // Optional per-token scale [total_M] float, empty = skip
    const at::Tensor& per_token_scale)
{
    using namespace v2;

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
    // Use uint8_t* base for arithmetic to avoid reinterpret_cast const issues
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

    auto* prob_dev   = reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
                           buf_problem_sizes_dev.data_ptr());
    auto* prob_host  = reinterpret_cast<const typename ProblemShape::UnderlyingProblemShape*>(
                           problem_sizes_host.data_ptr());

    // ------------------------------------------------------------------
    // Step 1: fill all CUTLASS args on device — zero host-device copies
    // ------------------------------------------------------------------
    v2::fill_grouped_gemm_args_kernel<<<1, num_groups, 0, stream>>>(
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
        K_packed, K, N
    );

    // ------------------------------------------------------------------
    // Step 2: build CUTLASS Arguments (all device pointers, no alloc)
    // ------------------------------------------------------------------
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id              = A.device().index();
    hw_info.sm_count               = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    hw_info.cluster_shape          = dim3(2, 1, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);

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
            {},
            ptr_C,  stride_C,
            ptr_D,  stride_D
        },
        hw_info
    };
    arguments.epilogue.thread.alpha = ElementAccumulator(1);
    arguments.epilogue.thread.beta  = ElementAccumulator(0);

    // ------------------------------------------------------------------
    // Step 3: initialize + run (workspace reused, no cudaMalloc)
    // ------------------------------------------------------------------
    Gemm gemm;
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
    CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));

    // Optional per-token scale (same stream, no sync needed)
    if (per_token_scale.defined() && per_token_scale.numel() > 0) {
        const int total_M = static_cast<int>(D.size(0));
        const int N_out   = static_cast<int>(D.size(1));
        const int threads = std::min(N_out, 128);
        per_token_scale_kernel_v2<v2::ElementD><<<total_M, threads, 0, stream>>>(
            static_cast<v2::ElementD*>(D.data_ptr()),
            static_cast<const float*>(per_token_scale.data_ptr()),
            N_out);
    }
}
