/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "../util/multi_stream.h"
#include "./config.h"
#include "./cutlass_grouped_gemm.cuh"

namespace {

/* Use CUDA const memory to store scalar 1 and 0 for cublas usage
*/
__device__ __constant__ float one_device;
__device__ __constant__ float zero_device;

inline float *GetScalarOne() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float one = 1.0f;
    NVTE_CHECK_CUDA(cudaMemcpyToSymbol(one_device, &one, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float *dev_ptr;
  NVTE_CHECK_CUDA(cudaGetSymbolAddress(reinterpret_cast<void **>(&dev_ptr), one_device));
  return dev_ptr;
}

inline float *GetScalarZero() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float zero = 0.0f;
    NVTE_CHECK_CUDA(cudaMemcpyToSymbol(zero_device, &zero, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float *dev_ptr;
  NVTE_CHECK_CUDA(cudaGetSymbolAddress(reinterpret_cast<void **>(&dev_ptr), zero_device));
  return dev_ptr;
}

__global__ __launch_bounds__(1) void set_float_kernel(float *ptr, float val) { *ptr = val; }

uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (address % alignment == 0) {
      return alignment;
    }
  }
}

inline void CreateCublasHandle(cublasLtHandle_t *handle) {
  NVTE_CHECK_CUBLAS(cublasLtCreate(handle));
}

void cublas_gemm(const Tensor *inputA,
                 const Tensor *A_scale,
                 const Tensor *inputB,
                 const Tensor *B_scale,
                 Tensor *outputD,
                 cublasOperation_t transA,
                 cublasOperation_t transB, 
                 void *workspace, 
                 size_t workspaceSize,
                 const void *alpha, 
                 const void *beta, 
                 bool use_split_accumulator, 
                 int math_sm_count,
                 cudaStream_t stream) {
  // Tensor dims in row-major order
  const int A0 = inputA->flat_first_dim();
  const int A1 = inputA->flat_last_dim();
  const int B0 = inputB->flat_first_dim();
  const int B1 = inputB->flat_last_dim();

  // GEMM dims in column-major order
  const int m = transA == CUBLAS_OP_T ? A0 : A1;
  const int n = transB == CUBLAS_OP_T ? B1 : B0;
  const int k = transA == CUBLAS_OP_T ? A1 : A0;
  TORCH_CHECK((transB == CUBLAS_OP_T ? B0 : B1) == k,
             "GEMM inputs have incompatible dimensions (A is ", A0, "x", A1, ", B is ", B0, "x", B1,
             ")");
  const int ldd = m;

  // Return immediately if GEMM is trivial
  if (m <= 0 || n <= 0) {
    return;
  }
  TORCH_CHECK(k > 0);

  using cublasHandleManager = HandleManager<cublasLtHandle_t, CreateCublasHandle>;
  cublasLtHandle_t handle = cublasHandleManager::Instance().GetHandle();
  
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

  int64_t ld_gelumat = (int64_t)ldd;

  const cudaDataType_t A_type = CUDA_R_4F_E2M1;
  const cudaDataType_t B_type = CUDA_R_4F_E2M1;
  const cudaDataType_t D_type = CUDA_R_16BF;
  if (inputB->dtype() == Half) {
    D_type = CUDA_R_16F;
  } else if (inputB->dtype() == Float) {
    D_type = CUDA_R_32F;
  }

  // Use TF32 only for pure FP32 GEMM.
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  if (A_type == CUDA_R_32F && B_type == CUDA_R_32F && D_type == CUDA_R_32F) {
    gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  // First, set input and output type 
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type, transA == CUBLAS_OP_N ? m : k,
                                               transA == CUBLAS_OP_N ? k : m, param.lda));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type, transB == CUBLAS_OP_N ? k : n,
                                               transB == CUBLAS_OP_N ? n : k, param.ldb));

  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &transA, sizeof(transA)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                   &transB, sizeof(transB)));
  // Set math SM count
  if (math_sm_count != 0) {
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                     CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                     &math_sm_count, sizeof(math_sm_count)));
  }

  // Fast accumulation is only supported for FP8.
  const int8_t fastAccuMode = (use_split_accumulator) ? 0 : 1;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                                                    &fastAccuMode, sizeof(fastAccuMode)));

  // Scaling factors.
  cublasLtMatmulMatrixScale_t scaling_mode_a;
  cublasLtMatmulMatrixScale_t scaling_mode_b;
  
  // make sure alpha beta computation dtype remains fp32 by CUBLASLT_MATMUL_DESC_SCALE_TYPE
  cublasDataType_t scale_type = CUDA_R_32F;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

  // Set pointer mode: alpha and beta are both device pointers
  // https://docs.nvidia.com/cuda/cublas/#cublasltpointermode-t
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

  fp8e4m3 *A_scale_inverse = reinterpret_cast<fp8e4m3 *>(A_scale->data());
  fp8e4m3 *B_scale_inverse = reinterpret_cast<fp8e4m3 *>(B_scale->data());
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &A_scale_inverse, sizeof(A_scale_inverse)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &B_scale_inverse, sizeof(B_scale_inverse)));
  scaling_mode_a = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  scaling_mode_b = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                       CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                       &scaling_mode_a, sizeof(scaling_mode_a)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                    &scaling_mode_b, sizeof(scaling_mode_b)));
  
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));

  // set epilogue
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue)));

  // align the workspace to 256 B
  const int required_alignment = 256;
  const auto original_workspace_alignment = _getAlignment(reinterpret_cast<uintptr_t>(workspace));
  uint8_t *aligned_workspace_ptr =
      reinterpret_cast<uint8_t *>(workspace) + required_alignment - original_workspace_alignment;
  workspaceSize = workspaceSize - required_alignment + original_workspace_alignment;
  const auto new_workspace_alignment =
      _getAlignment(reinterpret_cast<uintptr_t>(aligned_workspace_ptr));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
  const auto A_alignment = _getAlignment(reinterpret_cast<uintptr_t>(inputA->data()));
  const auto B_alignment = _getAlignment(reinterpret_cast<uintptr_t>(inputB->data()));
  const auto C_alignment = _getAlignment(reinterpret_cast<uintptr_t>(outputD->data()));
  const auto D_alignment = _getAlignment(reinterpret_cast<uintptr_t>(outputD->data()));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &A_alignment, sizeof(A_alignment)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &B_alignment, sizeof(B_alignment)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &C_alignment, sizeof(C_alignment)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &D_alignment, sizeof(D_alignment)));
  NVTE_CHECK(new_workspace_alignment % 256 == 0,
             "cuBLAS workspace pointer must be aligned to 256 bytes, got ",
             new_workspace_alignment);

  const auto status =
      cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
                                     1, &heuristicResult, &returnedResults);
  NVTE_CHECK(status != CUBLAS_STATUS_NOT_SUPPORTED,
             "Unable to find suitable cuBLAS GEMM algorithm");
  NVTE_CHECK_CUBLAS(status);
  if (returnedResults == 0) NVTE_ERROR("Unable to find any suitable algorithms");


  // D = alpha * (A * B) + beta * C
  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle, operationDesc, alpha, /* alpha */
                                   inputA->data(),                      /* A */
                                   Adesc, inputB->data(),               /* B */
                                   Bdesc, beta,                  /* beta */
                                   outputD->data(),                            /* C */
                                   Cdesc, outputD->data(),                     /* D */
                                   Ddesc, &heuristicResult.algo, /* algo */
                                   aligned_workspace_ptr,        /* workspace */
                                   workspaceSize, stream));      /* stream */

  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}

}  // namespace transformer_engine

void nvte_cublas_gemm_v2( 
  const at::Tensor A,
  bool transa,
  const at::Tensor B, 
  bool transb,
  float alpha,
  std::optional<float> beta, 
  at::Tensor D,
  const at::Tensor workspace_tensor,
  size_t workspace_size,
) {

  auto stream = at::cuda::getCurrentCUDAStream();

  // T是rowwise, N是colwise
  // Optionally swizzle the scaling factors
  swizzled_scale_inverses_list.emplace_back(std::move(swizzle_scaling_factors(A_tensor, transa)));
  swizzled_scale_inverses_list.emplace_back(
  std::move(swizzle_scaling_factors(B_tensor, !transb)));

  transa = true;
  transb = false;


  // Launch GEMM
  cublas_gemm(
    A_tensor,
    A_scale,
    B_tensor,
    B_scale,
    D_tensor,
    transa ? CUBLAS_OP_T : CUBLAS_OP_N, 
    transb ? CUBLAS_OP_T : CUBLAS_OP_N,
    workspace_tensor->data(), 
    workspace_size, 
    &alpha, &beta.value(),
    config_.use_split_accumulator, 
    config_.sm_count, 
    stream
  );
}