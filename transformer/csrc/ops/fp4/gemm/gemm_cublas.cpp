#include <pybind11/pybind11.h>

#include <optional>
#include <string>

#include "../common.h"
#include "../extensions.h"
#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"
#include "util.h"

size_t product(const std::vector<size_t>& shape, size_t begin, size_t end) {
  TORCH_CHECK(begin <= end && end <= shape.ndim, "Attempted to access entries ", begin, " to ", end,
             " in a shape with ", shape.ndim, " entries");
  size_t ret = 1;
  for (size_t i = begin; i < end; ++i) {
    ret *= shape[i];
  }
  return ret;
}

std::vector<size_t> getGemmOutputShape(const std::vector<size_t>& A_shape, const bool transa,
                                       const std::vector<size_t>& B_shape, const bool transb) {
  // Flatten outer dims to get 2D matrices
  const size_t A0 = A_shape.size() > 0 ? product(A_shape, 0, A_shape.size() - 1) : 1;
  const size_t A1 = A_shape.size() > 0 ? A_shape[A_shape.size() - 1] : 1;
  const size_t B0 = B_shape.size() > 0 ? product(B_shape, 0, B_shape.size() - 1) : 1;
  const size_t B1 = B_shape.size() > 0 ? B_shape[B_shape.size() - 1] : 1;

  // Check matrix dims
  TORCH_CHECK((transa ? A1 : A0) == (transb ? B0 : B1), "Invalid matrix dimensions for GEMM (A=(",
             A0, ",", A1, "), transa=", transa, ", B=(", B0, ",", B1, "), transb=", transb, ")");

  // Construct output dims
  std::vector<size_t> ret;
  if (transb) {
    ret.emplace_back(B1);
  } else {
    // Unflatten B0
    for (size_t i = 0; i < B_shape.ndim - 1; ++i) {
      ret.emplace_back(B_shape.data[i]);
    }
  }
  if (transa) {
    ret.emplace_back(A0);
  } else {
    ret.emplace_back(A1);
  }
  return ret;
}

std::vector<py::object> gemm(at::Tensor A_tensor,
                             at::Tensor A_scale,
                             bool transa,
                             at::Tensor B_tensor,
                             at::Tensor B_scale,
                             bool transb,
                             at::Tensor D_tensor,
                             std::optional<at::ScalarType> out_dtype,
                             at::Tensor bias,
                             std::optional<at::ScalarType> bias_type,
                             bool gelu, MaybeTensor gelu_in, bool grad,
                             at::Tensor workspace, size_t workspaceSize, bool accumulate,
                             bool use_split_accumulator, CommOverlapCore* comm_overlap,
                             std::optional<CommOverlapType> comm_type, MaybeTensor extra_output,
                             bool bulk_overlap, float alpha, std::optional<float> beta) {

  // Ensure that cublasLt handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(workspace.device());

  // Input tensors
  TORCH_CHECK(!A.is_none(), "Tensor A has not been provided");
  TORCH_CHECK(!B.is_none(), "Tensor B has not been provided");

  const bool low_precision =
      detail::is_low_precision(A_tensor.dtype()) || detail::is_low_precision(B_tensor.dtype());
  const bool fp8_block_scaling = A_tensor.scaling_mode() == NVTE_BLOCK_SCALING_1D ||
                                 A_tensor.scaling_mode() == NVTE_BLOCK_SCALING_2D ||
                                 B_tensor.scaling_mode() == NVTE_BLOCK_SCALING_1D ||
                                 B_tensor.scaling_mode() == NVTE_BLOCK_SCALING_2D;

  // Check tensor dimensions
  const auto& A_shape = A_tensor.sizes();
  const auto& B_shape = B_tensor.sizes();
  const auto& D_shape = getGemmOutputShape(A_shape, transa, B_shape, transb);
  TORCH_CHECK(A_shape.size() >= 1, "Tensor A needs to have at least 1 dimension");
  TORCH_CHECK(B_shape.size() >= 1, "Tensor B needs to have at least 1 dimension");

  // Check scaling factors
  if (accumulate) {
    if (!beta) {
      beta = 1.0f;
    }
  } else {
    if (!beta) {
      beta = 0.0f;
    }
    TORCH_CHECK(beta == 0.0, "Trying to use non-zero beta while not accumulating ",
               "into D tensor. Beta has nothing to be applied to.");
  }

  auto output_dtype = out_dtype ? *out_dtype : A_tensor.scalar_type();
  auto opts = A_tensor.options();
  torch::empty(D_shape, opts.dtype(output_dtype));


  // Set an external SM Margin to all the GEMMs.
  // This comes in handy when DP is overlapped with GEMMs
  const int device_id = at::cuda::current_device();
  const int sm_count = sm_count(device_id);

  // Construct GEMM config
  transformer_engine::MatmulConfigWrapper config;
  if (grad) {
    config.set_dbias_tensor(bias_tensor.data());
  } else {
    config.set_bias_tensor(bias_tensor.data());
  }
  config.set_use_split_accumulator(use_split_accumulator);
  config.set_sm_count(sm_count);

  // Keep the swizzled scaling factor tensors alive during the GEMM.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;
  auto main_stream = at::cuda::getCurrentCUDAStream();

  if (A_tensor.numel() != 0 && B_tensor.numel() != 0) {
    // Optionally swizzle the scaling factors
    swizzled_scale_inverses_list.emplace_back(std::move(swizzle_scaling_factors(A_tensor, transa)));
    swizzled_scale_inverses_list.emplace_back(
        std::move(swizzle_scaling_factors(B_tensor, !transb)));

    // Emulate the FP8 block scaling recipe with MXFP8 on Blackwell and newer
    // as it is not natively supported by cublasLt
    if (fp8_block_scaling && transformer_engine::cuda::sm_arch() >= 100) {
      // Convert tensors to mxfp8 and swizzle their scaling factors
      swizzled_scale_inverses_list.emplace_back(
          std::move(convert_block_scaling_to_mxfp8_tensor(A_tensor, transa)));
      swizzled_scale_inverses_list.emplace_back(
          std::move(convert_block_scaling_to_mxfp8_tensor(B_tensor, !transb)));
      // Use TN GEMM to avoid having to transpose data.
      transa = true;
      transb = false;
    }
    // Launch GEMM
    NVTE_SCOPED_GIL_RELEASE({
      nvte_cublas_gemm_v2(transa, transb, &alpha, A_tensor.data(), B_tensor.data(), &beta.value(),
                          out_tensor.data(), out_tensor.data(), te_workspace.data(), config,
                          main_stream);
    });
  } else {
    if (out_tensor.numel() != 0 && !accumulate) {
      out_tensor.zero_(main_stream);
    }
    if (bias.has_value()) {
      if (bias->numel() != 0 && grad) {
        bias_grad->zero_();
      }
    }
  }
  if (unfused_quantization_needed) {
    // Quantize the output
    std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
    my_quantizer->quantize(unquantized_D_tensor, D_tensor);
  }
  // Pack outputs
  std::vector<py::object> out;
  out.emplace_back(std::move(D));
  out.emplace_back(py::cast(bias_grad));
  if (gelu && !grad) {
    out.emplace_back(py::cast(*pre_gelu_out));
  } else {
    out.emplace_back(py::none());
  }
  if (extra_output.has_value()) {
    out.emplace_back(py::cast(extra_output));
  } else {
    out.emplace_back(py::none());
  }
  return out;
}