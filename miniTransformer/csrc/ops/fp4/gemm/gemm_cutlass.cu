/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef MINI_TE_NO_BLACKWELL_OPS

#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "../../../ops.h"
#include "../../../util/transformer_engine/common.h"

// Helper macro for checking CUTLASS errors
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

using namespace cute;

template <typename MmaTileShape, typename ClusterShape, typename PerSmTileShape_MNK,
          typename ArchTag,
          typename ElementA, typename LayoutATag, int AlignmentA,
          typename ElementB, typename LayoutBTag, int AlignmentB>
struct FpGemm {
    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementAccumulator = float;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            PerSmTileShape_MNK, ClusterShape,
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
                static_cast<int>(
                    sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::collective::KernelScheduleAuto
            >::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<
            Shape<int, int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue,
            void>;

    using Gemm =
        cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm, typename ScaleType>
typename Gemm::Arguments args_from_options(
                                at::Tensor& D,
                                at::Tensor const& A,
                                at::Tensor const& B,
                                at::Tensor const& A_sf,
                                at::Tensor const& B_sf,
                                torch::Tensor const& alpha,
                                int M, int N, int K)
{
    using ElementA       = typename Gemm::ElementA;
    using ElementB       = typename Gemm::ElementB;
    using ElementD       = typename Gemm::ElementD;
    using ElementSFA     = ScaleType;
    using ElementSFB     = ScaleType;
    using ElementCompute = float;
    using ElementAccumulator = float;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    using Sm1xxBlkScaledConfig =
        typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<ElementA const*>(A.data_ptr()),      stride_A,
            static_cast<ElementB const*>(B.data_ptr()),      stride_B,
            static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
            static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
        {
            {},
            static_cast<ElementD const*>(D.data_ptr()), stride_D,
            static_cast<ElementD*>(D.data_ptr()),       stride_D
        }
    };
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<ElementAccumulator const*>(alpha.data_ptr());

    return arguments;
}

template <typename Gemm, typename ScaleType>
void runGemm(at::Tensor& D,
             at::Tensor const& A,
             at::Tensor const& B,
             at::Tensor const& A_sf,
             at::Tensor const& B_sf,
             torch::Tensor const& alpha,
             int M, int N, int K,
             torch::Device device)
{
    Gemm gemm;

    auto arguments =
        args_from_options<Gemm, ScaleType>(D, A, B, A_sf, B_sf, alpha, M, N, K);

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());

    CUTLASS_CHECK(gemm.can_implement(arguments));

    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get(), stream));

    CUTLASS_CHECK(gemm.run(arguments, workspace.get(), stream));
}

void cutlass_gemm(at::Tensor& D,
                              at::Tensor const& A,
                              at::Tensor const& B,
                              at::Tensor const& A_sf,
                              at::Tensor const& B_sf,
                              at::Tensor const& alpha)
{
    auto const m = A.sizes()[0];
    auto const n = B.sizes()[0];
    auto const k = A.sizes()[1] * 2;

    using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;

    using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ArchTag = cutlass::arch::Sm100;
    if(m<=16){
        using MmaTileShape       = Shape<_128,_128,_256>;
        using ClusterShape       = Shape<_1,_1,_1>;
        using PerSmTileShape_MNK = Shape<_128,_128,_256>;
        runGemm<FpGemm<MmaTileShape, ClusterShape, PerSmTileShape_MNK,
                        ArchTag,
                        ElementA, LayoutATag, AlignmentA,
                        ElementB, LayoutBTag, AlignmentB>::Gemm, cutlass::float_ue4m3_t
                    >(D, A, B, A_sf, B_sf, alpha, m, n, k, A.device());
    } else if(m<=256){
        using MmaTileShape       = Shape<_256,_128,_256>;
        using ClusterShape       = Shape<_2,_1,_1>;
        using PerSmTileShape_MNK = Shape<_128,_128,_256>;
        runGemm<FpGemm<MmaTileShape, ClusterShape, PerSmTileShape_MNK,
                        ArchTag,
                        ElementA, LayoutATag, AlignmentA,
                        ElementB, LayoutBTag, AlignmentB>::Gemm, cutlass::float_ue4m3_t
                    >(D, A, B, A_sf, B_sf, alpha, m, n, k, A.device());
    } else {
        using MmaTileShape       = Shape<_256,_256,_256>;
        using ClusterShape       = Shape<_2,_1,_1>;
        using PerSmTileShape_MNK = Shape<_128,_256,_256>;
        runGemm<FpGemm<MmaTileShape, ClusterShape, PerSmTileShape_MNK,
                        ArchTag,
                        ElementA, LayoutATag, AlignmentA,
                        ElementB, LayoutBTag, AlignmentB>::Gemm, cutlass::float_ue4m3_t
                    >(D, A, B, A_sf, B_sf, alpha, m, n, k, A.device());
    }

}

#else  // MINI_TE_NO_BLACKWELL_OPS — stubs for non-Blackwell builds

#include "../../../ops.h"
#include "../../../util/transformer_engine/logging.h"

void cutlass_gemm(
  at::Tensor& D,
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& A_sf,
  const at::Tensor& B_sf,
  const at::Tensor& alpha
) {
  NVTE_ERROR("cutlass_gemm requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

#endif  // MINI_TE_NO_BLACKWELL_OPS
