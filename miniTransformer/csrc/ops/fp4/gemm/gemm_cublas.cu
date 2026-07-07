/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef MINI_TE_NO_BLACKWELL_OPS

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "../../handle_manager.h"
#include "../../../util/transformer_engine/common.h"
#include "../../swizzle.h"

inline void CreateCublasHandle(cublasLtHandle_t *handle) {
  NVTE_CHECK_CUBLAS(cublasLtCreate(handle));
}


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
  if (outputD->dtype() == at::ScalarType::Half) {
    D_type = CUDA_R_16F;
  } else if (outputD->dtype() == at::ScalarType::Float) {
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
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, gemm_compute_type, CUDA_R_32F));

  // Create matrix descriptors. Not setting any extra attributes.
  // First, set input and output type 
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, A_type, transA == CUBLAS_OP_N ? m : k,
                                               transA == CUBLAS_OP_N ? k : m, lda));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, B_type, transB == CUBLAS_OP_N ? k : n,
                                               transB == CUBLAS_OP_N ? n : k, ldb));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, m, n, ldd));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Ddesc, D_type, m, n, ldd));
  
  // Set operation attributes
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
  
  // make sure alpha beta computation dtype remains fp32 by CUBLASLT_MATMUL_DESC_SCALE_TYPE
  // Set scale type
  cublasDataType_t scale_type = CUDA_R_32F;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

  // Set pointer mode: alpha and beta are both device pointers
  // https://docs.nvidia.com/cuda/cublas/#cublasltpointermode-t
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

  // Set scaling mode.
  cublasLtMatmulMatrixScale_t scaling_mode_a = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  cublasLtMatmulMatrixScale_t scaling_mode_b = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                       CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                       &scaling_mode_a, sizeof(scaling_mode_a)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                    &scaling_mode_b, sizeof(scaling_mode_b)));
  
  // Set scale pointers
  void *A_scale_inverse = reinterpret_cast<void *>(A_scale->data_ptr());
  void *B_scale_inverse = reinterpret_cast<void *>(B_scale->data_ptr());
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &A_scale_inverse, sizeof(A_scale_inverse)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &B_scale_inverse, sizeof(B_scale_inverse)));
  
  // Create preference
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
  
  // set epilogue
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue)));

  const auto status =
      cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
                                     1, &heuristicResult, &returnedResults);
  NVTE_CHECK(status != CUBLAS_STATUS_NOT_SUPPORTED,
             "Unable to find suitable cuBLAS GEMM algorithm");
  NVTE_CHECK_CUBLAS(status);
  if (returnedResults == 0) throw std::runtime_error("Unable to find any suitable algorithms");

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
inline float compute_nvfp4_per_tensor_scale_kernel(float alpha_in, const float amax_A, const float amax_B) {
  // factor is defined in the enclosing namespace
  float alpha_out = alpha_in * (amax_A) * (amax_B) * factor_inv;
  return alpha_out;
}

// GEMM for NVFP4: D = (A * scale_A) @ (B * scale_B)
// A: (M, K//2) uint8 packed FP4 (2 elements per byte)
// B: (N, K//2) uint8 packed FP4 (2 elements per byte)  
// scale_A: FP8 E4M3 scales for A
// scale_B: FP8 E4M3 scales for B
// D: (M, N) BP16 
void cublas_gemm( 
  const at::Tensor A,
  const at::Tensor A_scale,
  float amax_A,
  bool transa,
  const at::Tensor B,
  const at::Tensor B_scale,
  float amax_B,
  bool transb,
  float alpha,
  std::optional<float> beta, 
  at::Tensor D,
  bool accumulate,
  const at::Tensor workspace_tensor,
  size_t workspace_size
) {

  auto stream = at::cuda::getCurrentCUDAStream();

  // T是rowwise, N是colwise
  // Optionally swizzle the scaling factors
  at::Tensor A_scale_swizzle = swizzle_scaling_factors(A_scale, transa);
  at::Tensor B_scale_swizzle = swizzle_scaling_factors(B_scale, !transb);

  // Compute alpha
  float new_alpha = 1.0f;
  new_alpha = compute_nvfp4_per_tensor_scale_kernel(alpha, amax_A, amax_B);

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
    static_cast<const void*>(&new_alpha), accumulate,
    true, 
    num_math_sms, 
    stream
  );
}

// Block-scaled matmul for NVFP4: D = (A * scale_A) @ (B * scale_B)
// A: (M, K//2) uint8 packed FP4 (2 elements per byte)
// B: (N, K//2) uint8 packed FP4 (2 elements per byte)  
// scale_A: FP8 E4M3 scales for A
// scale_B: FP8 E4M3 scales for B
// output: (M, N) FP16
void block_scaled_matmul_nvfp4_impl(
    const at::Tensor *A,
    const at::Tensor *B,
    at::Tensor *output,
    const at::Tensor *scale_A,
    const at::Tensor *scale_B,
    void *workspace,
    size_t workspaceSize,
    float *alpha,
    bool accumulate,
    cudaStream_t stream) {
  const int M = A->size(0);
  const int K_bytes = A->size(1);  // K in bytes
  const int K = K_bytes * 2;       // K in elements (2 elements per byte)
  const int N = B->size(0);
  
  TORCH_CHECK(B->size(1) == K_bytes, "B K dimension mismatch: expected ", K_bytes, " bytes, got ", B->size(1));
  TORCH_CHECK(A->dtype() == at::ScalarType::Byte, "A must be uint8, got ", A->dtype());
  TORCH_CHECK(B->dtype() == at::ScalarType::Byte, "B must be uint8, got ", B->dtype());

  using cublasHandleManager = HandleManager<cublasLtHandle_t, CreateCublasHandle>;
  cublasLtHandle_t handle = cublasHandleManager::Instance().GetHandle();

  cublasLtMatmulDesc_t matmulDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // Use FP32 compute and accumulation
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
  
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  // Set scale mode for NVFP4: 16-element groups with FP8 E4M3 scales
  cublasLtMatmulMatrixScale_t ab_scale_type = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ab_scale_type,
      sizeof(ab_scale_type)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &ab_scale_type,
      sizeof(ab_scale_type)));

  // Set scale pointers
  void *scale_A_ptr = const_cast<void*>(scale_A->data_ptr());
  void *scale_B_ptr = const_cast<void*>(scale_B->data_ptr());
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_A_ptr,
      sizeof(scale_A_ptr)));
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_B_ptr,
      sizeof(scale_B_ptr)));
  // Create matrix layouts
  // NVFP4: CUDA_R_4F_E2M1
  // With transa=T: A layout is (k, m), lda=k
  // With transb=N: B layout is (k, n), ldb=k
  cudaDataType_t dataType = CUDA_R_4F_E2M1;
  cudaDataType_t D_type = CUDA_R_16BF;
  if (output->dtype() == at::ScalarType::Half) {
    D_type = CUDA_R_16F;
  } else if (output->dtype() == at::ScalarType::Float) {
    D_type = CUDA_R_32F;
  }

  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, dataType, K, M, K));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, dataType, K, N, K));
  NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, D_type, M, N, M));

  float* beta = (accumulate) ? GetScalarOne() : GetScalarZero();

  // Create preference
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
      sizeof(workspaceSize)));

  // Set pointer mode: alpha and beta are both device pointers
  // https://docs.nvidia.com/cuda/cublas/#cublasltpointermode-t
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  NVTE_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

  // Query cuBLAS heuristics for the best algorithm
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      handle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
      &heuristicResult, &returnedResults);

  if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
    NVTE_ERROR("cublasLtMatmulAlgoGetHeuristic failed (status=",
               cublasGetStatusString(status), ", results=", std::to_string(returnedResults), ") for nvfp4");
  }

  // Execute matmul with the selected algorithm
  // B and A are swapped for row-major to col-major conversion
  NVTE_CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, alpha,
                                   const_cast<void*>(A->data_ptr()), Adesc,
                                   const_cast<void*>(B->data_ptr()), Bdesc,
                                   beta,
                                   output->data_ptr(), Cdesc,
                                   output->data_ptr(), Cdesc, 
                                   &heuristicResult.algo,
                                   workspace, workspaceSize, stream));

  // Cleanup
  NVTE_CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc)
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc)
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    NVTE_CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
  if (matmulDesc)
    NVTE_CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
}

void block_scaled_matmul_nvfp4(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& output,
    const at::Tensor& scale_A,
    const at::Tensor& scale_B,
    float amax_A,
    float amax_B,
    at::Tensor& alpha,
    bool accumulate,
    const at::Tensor& workspace_tensor,
    size_t workspace_size) {
  auto stream = at::cuda::getCurrentCUDAStream();
  
  // Validate inputs
  TORCH_CHECK(A.dtype() == at::ScalarType::Byte, 
              "block_scaled_matmul_nvfp4 only supports uint8 packed FP4 inputs (NVFP4), got A dtype: ", A.dtype());
  TORCH_CHECK(B.dtype() == at::ScalarType::Byte,
              "block_scaled_matmul_nvfp4 only supports uint8 packed FP4 inputs (NVFP4), got B dtype: ", B.dtype());
  
  // For packed FP4, shape[1] is in bytes, but K dimension should be in elements
  // So K = A_shape[1] * 2 (2 elements per byte)
  int K = A.size(1) * 2;
  TORCH_CHECK(B.size(1) * 2 == K, 
              "K dimensions must match. A has ", K, " elements, B has ", B.size(1) * 2, " elements");

  // Compute alpha
  // float new_alpha = 1.0f;
  // new_alpha = compute_nvfp4_per_tensor_scale_kernel(alpha, amax_A, amax_B);

  block_scaled_matmul_nvfp4_impl(
      &A, &B, &output, &scale_A, &scale_B,
      workspace_tensor.data_ptr(), workspace_size, static_cast<float*>(alpha.data_ptr()), accumulate, stream);
}

#else  // MINI_TE_NO_BLACKWELL_OPS — stubs for non-Blackwell builds

#include "../../../ops.h"
#include <stdexcept>

void cublas_gemm(
  const at::Tensor A,
  const at::Tensor A_scale,
  float amax_A,
  bool transa,
  const at::Tensor B,
  const at::Tensor B_scale,
  float amax_B,
  bool transb,
  float alpha,
  std::optional<float> beta,
  at::Tensor D,
  bool accumulate,
  const at::Tensor workspace_tensor,
  size_t workspace_size
) {
  throw std::runtime_error("cublas_gemm requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

void block_scaled_matmul_nvfp4(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& output,
    const at::Tensor& scale_A,
    const at::Tensor& scale_B,
    float amax_A,
    float amax_B,
    at::Tensor& alpha,
    bool accumulate,
    const at::Tensor& workspace_tensor,
    size_t workspace_size) {
  throw std::runtime_error("block_scaled_matmul_nvfp4 requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

#endif  // MINI_TE_NO_BLACKWELL_OPS

