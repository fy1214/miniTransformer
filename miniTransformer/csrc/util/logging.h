/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <nvrtc.h>

#include "nccl.h"
// #include <cublasmp.h>

#include <iostream>
#include <stdexcept>
#include <string>

/* \brief Convert container to string */
template <typename T, typename = typename std::enable_if<!std::is_arithmetic<T>::value>::type,
          typename = decltype(std::declval<T>().begin())>
inline std::string to_string_like(const T &container) {
  std::string str;
  str.reserve(1024);  // Assume strings are <1 KB
  str += "(";
  bool first = true;
  for (const auto &val : container) {
    if (!first) {
      str += ",";
    }
    str += to_string_like(val);
    first = false;
  }
  str += ")";
  return str;
}

/*! \brief Convert arguments to strings and concatenate */
template <typename... Ts>
inline std::string concat_strings(const Ts &...args) {
  std::string str;
  str.reserve(1024);  // Assume strings are <1 KB
  (..., (str += to_string_like(args)));
  return str;
}

#define NVTE_WARN(...)                                            \
  do {                                                            \
    std::cerr << concat_strings(            \
        __FILE__ ":", __LINE__, " in function ", __func__, ": ",  \
        concat_strings(__VA_ARGS__), "\n"); \
  } while (false)

#define NVTE_ERROR(...)                                              \
  do {                                                               \
    throw ::std::runtime_error(concat_strings( \
        __FILE__ ":", __LINE__, " in function ", __func__, ": ",     \
        concat_strings(__VA_ARGS__)));         \
  } while (false)

#define NVTE_CHECK(expr, ...)                                        \
  do {                                                               \
    if (!(expr)) {                                                   \
      NVTE_ERROR("Assertion failed: " #expr ". ",                    \
                 concat_strings(__VA_ARGS__)); \
    }                                                                \
  } while (false)

#define NVTE_CHECK_CUDA(expr)                                                 \
  do {                                                                        \
    const cudaError_t status_NVTE_CHECK_CUDA = (expr);                        \
    if (status_NVTE_CHECK_CUDA != cudaSuccess) {                              \
      NVTE_ERROR("CUDA Error: ", cudaGetErrorString(status_NVTE_CHECK_CUDA)); \
    }                                                                         \
  } while (false)

#define NVTE_CHECK_CUBLAS(expr)                                                      \
  do {                                                                               \
    const cublasStatus_t status_NVTE_CHECK_CUBLAS = (expr);                          \
    if (status_NVTE_CHECK_CUBLAS != CUBLAS_STATUS_SUCCESS) {                         \
      NVTE_ERROR("cuBLAS Error: ", cublasGetStatusString(status_NVTE_CHECK_CUBLAS)); \
    }                                                                                \
  } while (false)

#define NVTE_CHECK_CUDNN(expr)                                                  \
  do {                                                                          \
    const cudnnStatus_t status_NVTE_CHECK_CUDNN = (expr);                       \
    if (status_NVTE_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {                      \
      NVTE_ERROR("cuDNN Error: ", cudnnGetErrorString(status_NVTE_CHECK_CUDNN), \
                 ". "                                                           \
                 "For more information, enable cuDNN error logging "            \
                 "by setting CUDNN_LOGERR_DBG=1 and "                           \
                 "CUDNN_LOGDEST_DBG=stderr in the environment.");               \
    }                                                                           \
  } while (false)

#define NVTE_CHECK_CUDNN_FE(expr)                                    \
  do {                                                               \
    const auto error = (expr);                                       \
    if (error.is_bad()) {                                            \
      NVTE_ERROR("cuDNN Error: ", error.err_msg,                     \
                 ". "                                                \
                 "For more information, enable cuDNN error logging " \
                 "by setting CUDNN_LOGERR_DBG=1 and "                \
                 "CUDNN_LOGDEST_DBG=stderr in the environment.");    \
    }                                                                \
  } while (false)

#define NVTE_CHECK_NVRTC(expr)                                                   \
  do {                                                                           \
    const nvrtcResult status_NVTE_CHECK_NVRTC = (expr);                          \
    if (status_NVTE_CHECK_NVRTC != NVRTC_SUCCESS) {                              \
      NVTE_ERROR("NVRTC Error: ", nvrtcGetErrorString(status_NVTE_CHECK_NVRTC)); \
    }                                                                            \
  } while (false)

#ifdef NVTE_WITH_CUBLASMP

#define NVTE_CHECK_CUBLASMP(expr)                             \
  do {                                                        \
    const cublasMpStatus_t status = (expr);                   \
    if (status != CUBLASMP_STATUS_SUCCESS) {                  \
      NVTE_ERROR("cuBLASMp Error: ", std::to_string(status)); \
    }                                                         \
  } while (false)

#endif  // NVTE_WITH_CUBLASMP

#define NVTE_CHECK_NCCL(expr)                                                 \
  do {                                                                        \
    const ncclResult_t status_NVTE_CHECK_NCCL = (expr);                       \
    if (status_NVTE_CHECK_NCCL != ncclSuccess) {                              \
      NVTE_ERROR("NCCL Error: ", ncclGetErrorString(status_NVTE_CHECK_NCCL)); \
    }                                                                         \
  } while (false)

#define NVTE_DEVICE_ERROR(message)                                                                 \
  do {                                                                                             \
    printf("%s:%d in function %s (thread (%d,%d,%d), block (%d,%d,%d)): %s\n", __FILE__, __LINE__, \
           __func__, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,    \
           (message));                                                                             \
    assert(0);                                                                                     \
  } while (false)

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_