/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <algorithm>
#include <cstdint>

#include "../../handle_manager.h"
#include "../../../util/common.h"
#include "../../swizzle.cuh"

inline void CreateCublasHandle(cublasLtHandle_t *handle) {
  NVTE_CHECK_CUBLAS(cublasLtCreate(handle));
}

void cublas_gemm_impl(
    const at::Tensor *inputA,
    const at::Tensor *A_scale,
    const at::Tensor *inputB,
    const at::Tensor *B_scale,
    at::Tensor *outputD,
    cublasOperation_t transA,
    cublasOperation_t transB, 
    void *workspace, 
    size_t workspaceSize,
    const void *alpha, 
    bool accumulate,
    bool use_split_accumulator, 
    int math_sm_count,
    cudaStream_t stream) {
  // Tensor dims in row-major order
  const int A0 = inputA->size(0);
  const int A1 = inputA->size(1);
  const int B0 = inputB->size(0);
  const int B1 = inputB->size(1);

  // GEMM dims in column-major order
  const int m = transA == CUBLAS_OP_T ? A0 : A1;
  const int n = transB == CUBLAS_OP_T ? B1 : B0;
  const int k = transA == CUBLAS_OP_T ? A1 : A0;
  TORCH_CHECK((transB == CUBLAS_OP_T ? B0 : B1) == k,
             "GEMM inputs have incompatible dimensions (A is ", A0, "x", A1, ", B is ", B0, "x", B1,
             ")");

  int lda = k, ldb = k, ldc = m, ldd = m;

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
  cudaDataType_t D_type = CUDA_R_16BF;
  if (inputB->dtype() == at::ScalarType::Half) {
    D_type = CUDA_R_16F;
  } else if (inputB->dtype() == at::ScalarType::Float) {
    D_type = CUDA_R_32F;
  }

  float one = 1.0;
  float zero = 0.0;
  float beta = (accumulate) ? one : zero;

  // Use TF32 only for pure FP32 GEMM.
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  if (A_type == CUDA_R_32F && B_type == CUDA_R_32F && D_type == CUDA_R_32F) {
    gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }

  // Create matrix descriptors. Not setting any extra attributes.
  // First, set input and output type 
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type, transA == CUBLAS_OP_N ? m : k,
                                               transA == CUBLAS_OP_N ? k : m, lda));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type, transB == CUBLAS_OP_N ? k : n,
                                               transB == CUBLAS_OP_N ? n : k, ldb));

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

  fp8e4m3 *A_scale_inverse = reinterpret_cast<fp8e4m3 *>(A_scale->data_ptr());
  fp8e4m3 *B_scale_inverse = reinterpret_cast<fp8e4m3 *>(B_scale->data_ptr());
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
  const auto A_alignment = _getAlignment(reinterpret_cast<uintptr_t>(inputA->data_ptr()));
  const auto B_alignment = _getAlignment(reinterpret_cast<uintptr_t>(inputB->data_ptr()));
  const auto C_alignment = _getAlignment(reinterpret_cast<uintptr_t>(outputD->data_ptr()));
  const auto D_alignment = _getAlignment(reinterpret_cast<uintptr_t>(outputD->data_ptr()));
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
                                   inputA->data_ptr(), Adesc,           /* A */
                                   inputB->data_ptr(), Bdesc,           /* B */ 
                                   static_cast<const void*>(&beta),     /* beta */
                                   outputD->data_ptr(), Cdesc,          /* C */
                                   outputD->data_ptr(), Ddesc,          /* D */
                                   &heuristicResult.algo, /* algo */
                                   aligned_workspace_ptr,        /* workspace */
                                   workspaceSize, stream));      /* stream */

  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Ddesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
}

// constexpr float factor = 6.0 * 6.0 * 448.0 * 448.0;
constexpr float factor_inv = 1.0 / (6.0 * 6.0 * 448.0 * 448.0);

// compute alpha *= amax_A * amax_B / factor
void compute_nvfp4_per_tensor_scale_kernel(float alpha_in, const float *amax_A,
                                           const float *amax_B, float *alpha_out) {
  // factor is defined in the enclosing namespace
  *alpha_out = alpha_in * (*amax_A) * (*amax_B) * factor_inv;
}

at::Tensor swizzle_scaling_factors(
  const at::Tensor& input,
  bool rowwise
) {
  NVTE_CHECK(input.dim() >= 2, "Wrong ndims for swizzle input shape.");

  std::vector<int64_t> output_size = {input.sizes()[0], input.sizes()[1]};
  auto options = at::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
  auto out = at::empty(output_size, options);

  // Launch kernel
  swizzle_scaling_factors_kernel(
    input, 
    &out, 
    at::cuda::getCurrentCUDAStream(), 
    NVTE_NVFP4_1D_SCALING
  );

  return out;
}

void cublas_gemm( 
  const at::Tensor A,
  const at::Tensor A_scale,
  const float *amax_A,
  bool transa,
  const at::Tensor B,
  const at::Tensor B_scale,
  const float *amax_B,
  bool transb,
  float alpha,
  std::optional<float> beta, 
  at::Tensor D,
  bool accumulate,
  bool use_split_accumulator,
  const at::Tensor workspace_tensor,
  size_t workspace_size
) {

  auto stream = at::cuda::getCurrentCUDAStream();

  // T是rowwise, N是colwise
  // Optionally swizzle the scaling factors
  at::Tensor A_scale_swizzle = swizzle_scaling_factors(A_scale, transa);
  at::Tensor B_scale_swizzle = swizzle_scaling_factors(B_scale, !transb);

  // Compute alpha
  compute_nvfp4_per_tensor_scale_kernel(alpha, amax_A, amax_B, &alpha);

  // Get the number of streaming multiprocessors
  const int device_id = cuda::current_device();
  const int num_math_sms = cuda::sm_count(device_id);

  transa = true;
  transb = false;

  // Launch GEMM
  cublas_gemm_impl(
    &A,
    &A_scale_swizzle,
    &B,
    &B_scale_swizzle,
    &D,
    transa ? CUBLAS_OP_T : CUBLAS_OP_N,
    transb ? CUBLAS_OP_T : CUBLAS_OP_N,
    workspace_tensor.data_ptr(), 
    workspace_size, 
    static_cast<const void*>(&alpha), accumulate,
    use_split_accumulator, 
    num_math_sms, 
    stream
  );
}