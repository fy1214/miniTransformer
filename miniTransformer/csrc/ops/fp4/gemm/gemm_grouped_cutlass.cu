/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

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
#include "cutlass/util/device_memory.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "../../../ops.h"

#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

using namespace cute;

// ---------------------------------------------------------------------------
// Per-token scale kernel: D[m][n] *= scale[m]
// One CTA per row: threads stride through N elements (coalesced access).
// scale[m] is broadcast within the warp — loaded once per CTA.
// ---------------------------------------------------------------------------
template <typename ElementD>
__global__ void per_token_scale_kernel(
    ElementD* __restrict__ D,
    const float* __restrict__ scale,
    int N)
{
    int m = blockIdx.x;
    float s = scale[m];
    ElementD* row = D + (int64_t)m * N;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        row[n] = static_cast<ElementD>(static_cast<float>(row[n]) * s);
    }
}

template <typename ElementD>
static void launch_per_token_scale(
    ElementD* D_ptr,
    const float* scale_ptr,
    int M, int N,
    cudaStream_t stream)
{
    // 128 threads per block; each block handles one row
    const int threads = std::min(N, 128);
    per_token_scale_kernel<ElementD><<<M, threads, 0, stream>>>(D_ptr, scale_ptr, N);
}

// ---------------------------------------------------------------------------
// CUTLASS Grouped GEMM type definitions (SM100, FP4 input, BF16 output)
// ---------------------------------------------------------------------------
using ElementA           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD           = cutlass::bfloat16_t;
using ElementC           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using ArchTag            = cutlass::arch::Sm100;
using LayoutATag         = cutlass::layout::RowMajor;
using LayoutBTag         = cutlass::layout::ColumnMajor;
using LayoutCTag         = cutlass::layout::RowMajor;
using LayoutDTag         = cutlass::layout::RowMajor;
using ElementSF          = cutlass::float_ue4m3_t;

static constexpr int AlignmentA = 32;
static constexpr int AlignmentB = 32;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using MmaTileShape  = Shape<_128, _128, _256>;
using ClusterShape  = Shape<_1, _1, _1>;

// Explicit PtrArray schedule for grouped GEMM on SM100 with FP4 block-scaled inputs.
// KernelScheduleAuto would select the non-array (single-GEMM) dispatch path because
// IsArrayOfPointersGemm is false for KernelScheduleAuto — we must be explicit here.
using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmBlockScaledSm100;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
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
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA  = typename Gemm::GemmKernel::StrideA;
using StrideB  = typename Gemm::GemmKernel::StrideB;
using StrideC  = typename Gemm::GemmKernel::StrideC;
using StrideD  = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = decltype(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(0, 0, 0, 0)));
using LayoutSFB = decltype(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
    cute::make_shape(0, 0, 0, 0)));

// ---------------------------------------------------------------------------
// Public entry point
//
// grouped_cutlass_gemm_with_per_token_scale
//
// Args (all lists have length == num_groups):
//   D_list        : output tensors, (M_i, N_i) bfloat16
//   A_list        : (M_i, K_i//2) uint8 packed FP4, row-major
//   B_list        : (N_i, K_i//2) uint8 packed FP4, col-major (already transposed)
//   A_sf_list     : FP8-E4M3 scale factors for A, CUTLASS swizzled layout
//   B_sf_list     : FP8-E4M3 scale factors for B, CUTLASS swizzled layout
//   per_token_scale_list : (M_i,) float32, one scale per output row
//
// per_token_scale is applied after GEMM as a separate kernel:
//   D[m][n] *= per_token_scale[m]
// ---------------------------------------------------------------------------
void grouped_cutlass_gemm_with_per_token_scale(
    std::vector<at::Tensor>& D_list,
    const std::vector<at::Tensor>& A_list,
    const std::vector<at::Tensor>& B_list,
    const std::vector<at::Tensor>& A_sf_list,
    const std::vector<at::Tensor>& B_sf_list,
    const std::vector<at::Tensor>& per_token_scale_list)
{
    const int groups = static_cast<int>(A_list.size());
    TORCH_CHECK(groups > 0, "grouped_cutlass_gemm: empty group list");
    TORCH_CHECK((int)B_list.size() == groups);
    TORCH_CHECK((int)A_sf_list.size() == groups);
    TORCH_CHECK((int)B_sf_list.size() == groups);
    TORCH_CHECK((int)D_list.size() == groups);
    TORCH_CHECK((int)per_token_scale_list.size() == groups);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A_list[0]));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A_list[0].device().index());

    // ------------------------------------------------------------------
    // Build host-side problem shapes and pointer/stride arrays
    // ------------------------------------------------------------------
    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host(groups);
    std::vector<StrideA>  stride_A_host(groups);
    std::vector<StrideB>  stride_B_host(groups);
    std::vector<LayoutSFA> layout_SFA_host;
    std::vector<LayoutSFB> layout_SFB_host;
    std::vector<StrideC>  stride_C_host(groups);
    std::vector<StrideD>  stride_D_host(groups);

    std::vector<const ElementA*> ptr_A_host(groups);
    std::vector<const ElementB*> ptr_B_host(groups);
    std::vector<const ElementSF*> ptr_SFA_host(groups);
    std::vector<const ElementSF*> ptr_SFB_host(groups);
    std::vector<const ElementC*> ptr_C_host(groups);
    std::vector<ElementD*>       ptr_D_host(groups);

    for (int i = 0; i < groups; ++i) {
        int M = static_cast<int>(A_list[i].size(0));
        int K = static_cast<int>(A_list[i].size(1)) * 2;  // packed FP4: 2 elems/byte
        int N = static_cast<int>(B_list[i].size(0));

        TORCH_CHECK(B_list[i].size(1) * 2 == K, "group ", i, ": K mismatch");
        TORCH_CHECK(D_list[i].size(0) == M && D_list[i].size(1) == N,
                    "group ", i, ": D shape mismatch");
        TORCH_CHECK(per_token_scale_list[i].size(0) == M,
                    "group ", i, ": per_token_scale length must equal M=", M);

        problem_sizes_host[i] = cute::make_shape(M, N, K);

        stride_A_host[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        stride_B_host[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        stride_C_host[i] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        stride_D_host[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        layout_SFA_host.push_back(
            Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1)));
        layout_SFB_host.push_back(
            Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1)));

        ptr_A_host[i]   = static_cast<const ElementA*>(A_list[i].data_ptr());
        ptr_B_host[i]   = static_cast<const ElementB*>(B_list[i].data_ptr());
        ptr_SFA_host[i] = static_cast<const ElementSF*>(A_sf_list[i].data_ptr());
        ptr_SFB_host[i] = static_cast<const ElementSF*>(B_sf_list[i].data_ptr());
        ptr_C_host[i]   = static_cast<const ElementC*>(D_list[i].data_ptr());  // C = D (beta=0)
        ptr_D_host[i]   = static_cast<ElementD*>(D_list[i].data_ptr());
    }

    // ------------------------------------------------------------------
    // Copy pointer/stride arrays to device
    // ------------------------------------------------------------------
    cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes_dev(groups);
    problem_sizes_dev.copy_from_host(problem_sizes_host.data());

    cutlass::DeviceAllocation<StrideA>  stride_A_dev(groups);
    cutlass::DeviceAllocation<StrideB>  stride_B_dev(groups);
    cutlass::DeviceAllocation<LayoutSFA> layout_SFA_dev(groups);
    cutlass::DeviceAllocation<LayoutSFB> layout_SFB_dev(groups);
    cutlass::DeviceAllocation<StrideC>  stride_C_dev(groups);
    cutlass::DeviceAllocation<StrideD>  stride_D_dev(groups);

    stride_A_dev.copy_from_host(stride_A_host.data());
    stride_B_dev.copy_from_host(stride_B_host.data());
    layout_SFA_dev.copy_from_host(layout_SFA_host.data());
    layout_SFB_dev.copy_from_host(layout_SFB_host.data());
    stride_C_dev.copy_from_host(stride_C_host.data());
    stride_D_dev.copy_from_host(stride_D_host.data());

    cutlass::DeviceAllocation<const ElementA*>  ptr_A_dev(groups);
    cutlass::DeviceAllocation<const ElementB*>  ptr_B_dev(groups);
    cutlass::DeviceAllocation<const ElementSF*> ptr_SFA_dev(groups);
    cutlass::DeviceAllocation<const ElementSF*> ptr_SFB_dev(groups);
    cutlass::DeviceAllocation<const ElementC*>  ptr_C_dev(groups);
    cutlass::DeviceAllocation<ElementD*>        ptr_D_dev(groups);

    ptr_A_dev.copy_from_host(ptr_A_host.data());
    ptr_B_dev.copy_from_host(ptr_B_host.data());
    ptr_SFA_dev.copy_from_host(ptr_SFA_host.data());
    ptr_SFB_dev.copy_from_host(ptr_SFB_host.data());
    ptr_C_dev.copy_from_host(ptr_C_host.data());
    ptr_D_dev.copy_from_host(ptr_D_host.data());

    // ------------------------------------------------------------------
    // Build CUTLASS Arguments and run Grouped GEMM
    // ------------------------------------------------------------------
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups, problem_sizes_dev.get(), problem_sizes_host.data()},
        {
            ptr_A_dev.get(),   stride_A_dev.get(),
            ptr_B_dev.get(),   stride_B_dev.get(),
            ptr_SFA_dev.get(), layout_SFA_dev.get(),
            ptr_SFB_dev.get(), layout_SFB_dev.get()
        },
        {
            {},                         // thread fusion args (alpha=1, beta=0 defaults)
            ptr_C_dev.get(), stride_C_dev.get(),
            ptr_D_dev.get(), stride_D_dev.get()
        }
    };
    // alpha=1, beta=0: D = 1 * acc + 0 * C
    arguments.epilogue.thread.alpha = ElementAccumulator(1);
    arguments.epilogue.thread.beta  = ElementAccumulator(0);

    Gemm gemm;
    CUTLASS_CHECK(gemm.can_implement(arguments));

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));
    CUTLASS_CHECK(gemm.run(arguments, workspace.get(), stream));

    // ------------------------------------------------------------------
    // Apply per-token scale per group (separate kernel, same stream)
    // ------------------------------------------------------------------
    for (int i = 0; i < groups; ++i) {
        int M = static_cast<int>(A_list[i].size(0));
        int N = static_cast<int>(B_list[i].size(0));
        ElementD* D_ptr     = static_cast<ElementD*>(D_list[i].data_ptr());
        const float* s_ptr  = static_cast<const float*>(per_token_scale_list[i].data_ptr());
        launch_per_token_scale(D_ptr, s_ptr, M, N, stream);
    }
}
