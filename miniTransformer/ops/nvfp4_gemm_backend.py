import functools
from typing import Union

import torch
import fp4_gemm
import torch.nn.functional as F

from flashinfer import mm_fp4
from triton.tools.tensor_descriptor import TensorDescriptor
from miniTransformer.ops.triton.gemm_tl import block_scaled_matmul
from miniTransformer.ops.nvfp4_qgemm import cutedsl_nvfp4_gemm_optimized, swizzle_scale


class GemmBackendBase:

    def swizzle_scale(self, scale_2d, target_M, target_K):
        """
        2D scale (M, K//16) -> 4D (M//128, K//64, 32, 16) for triton/cuBLAS.
        Inverse of pack_scale_from_4d.
        """
        pad_length = (128 - (scale_2d.size(0) % 128)) % 128
        # 如果长度已经是 128 的倍数，则不需要填充
        if pad_length > 0:
            x_scale_padded = F.pad(scale_2d, pad=(0, 0, 0, pad_length), mode='constant', value=0)
            target_M = x_scale_padded.shape[0]
        else:
            x_scale_padded = scale_2d
        num_chunk_m, num_chunk_k = target_M // 128, (target_K // 16) // 4
        u = x_scale_padded.reshape(num_chunk_m, 4, 32, num_chunk_k, 4)
        return u.permute(0, 3, 2, 1, 4).reshape(
            num_chunk_m, num_chunk_k, 32, 16
        )

    def gemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        grad: bool,
        accumulate: bool,
        layout: str,
        alpha: torch.Tensor,
        tensor_scale_x: torch.Tensor,
        tensor_scale_w: torch.Tensor,
        has_swzzile: bool,
    ) -> torch.Tensor:
        pass



class NVFP4CutlassGemmBackend(GemmBackendBase):

    def gemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        grad: bool,
        accumulate: bool,
        layout: str,
        alpha: torch.Tensor,
        tensor_scale_x: torch.Tensor,
        tensor_scale_w: torch.Tensor,
        has_swzzile: bool,
    ) -> torch.Tensor:
        device = qx.device
        M, K_bytes_a = qx.shape
        N, K_bytes_b = qw.shape
        assert (
            K_bytes_a == K_bytes_b
        ), f"K dimension mismatch between qx ({K_bytes_a}) and qw ({K_bytes_b})"
        K = 2 * K_bytes_a  # 2 FP4 values per byte

        if not has_swzzile:
            sx = self.swizzle_scale(sx, M, K).to(torch.float8_e4m3fn)
            sw = self.swizzle_scale(sw, N, K).to(torch.float8_e4m3fn)

        if (
            K > 0
            and tensor_scale_x is not None
            and tensor_scale_w is not None
        ):
            factor = 6.0 * 6.0 * 448.0 * 448.0
            partial_alpha = (tensor_scale_x * tensor_scale_w).to(torch.float32)
            alpha = partial_alpha.squeeze(-1) / factor

        y = torch.empty((M, N), device=device, dtype=out_dtype)
        fp4_gemm.cutlass_gemm(
            y,  # output
            qx,  # A: packed NVFP4
            qw,  # B: packed NVFP4
            sx,  # A scales
            sw,  # B scales
            alpha
        )
        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out.add_(y)
            return out
        else:
            y = y.to(out_dtype)
            return y


_NUM_MAX_UB_STREAMS = 3
_NUM_COMPUTE_STREAM = 4

def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        # 32 MiB for NVFP4 GEMM, plus additional 1024 B for alignment and misc scales
        return 32 * 1024 * 1024 + 1024
    return 4_194_304


@functools.lru_cache(maxsize=None)
def get_cublas_workspace(device: int, ub: bool, grouped_gemm: bool) -> torch.Tensor:
    """Returns workspace for cublas GEMM."""
    assert not (ub and grouped_gemm), "UB is unsupported for grouped GEMM."

    if ub:
        return torch.empty(
            get_cublas_workspace_size_bytes() * _NUM_MAX_UB_STREAMS,
            dtype=torch.uint8,
            device=device,
        )
    if grouped_gemm:
        _multi_stream_cublas_workspace = []
        for _ in range(_NUM_COMPUTE_STREAM):
            _multi_stream_cublas_workspace.append(
                torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)
            )
        return _multi_stream_cublas_workspace

    return torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)


class NVFP4CublasGemmBackend(GemmBackendBase):

    def gemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        grad: bool,
        accumulate: bool,
        layout: str,
        alpha: torch.Tensor,
        tensor_scale_x: torch.Tensor,
        tensor_scale_w: torch.Tensor,
        has_swzzile: bool,
    ):
        m, n, k = qx.shape[0], qw.shape[0], qx.shape[1] * 2

        workspace = get_cublas_workspace(torch.cuda.current_device(), False, False)
        if not accumulate:
            out = torch.empty((m, n), dtype=out_dtype, device=qx.device)

        if (
            k > 0
            and tensor_scale_x is not None
            and tensor_scale_w is not None
        ):
            factor = 6.0 * 6.0 * 448.0 * 448.0
            partial_alpha = (tensor_scale_x * tensor_scale_w).to(torch.float32)
            alpha = partial_alpha.squeeze(-1) / factor

        fp4_gemm.block_scaled_matmul_nvfp4(
            qw,
            qx,
            out,
            sw if has_swzzile else self.swizzle_scale(sw, n, k),
            sx if has_swzzile else self.swizzle_scale(sx, m, k),
            1.0,
            1.0,
            alpha,
            accumulate,
            workspace,
            workspace.shape[0]
        )

        return out


class NVFP4CuteDSLGemmBackend(GemmBackendBase):

    def gemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        grad: bool,
        accumulate: bool,
        layout: str,
        alpha: torch.Tensor,
        tensor_scale_x: torch.Tensor,
        tensor_scale_w: torch.Tensor,
        has_swzzile: bool,
    ) -> torch.Tensor:
        device = qx.device
        M, K_bytes_a = qx.shape
        N, K_bytes_b = qw.shape
        assert (
                K_bytes_a == K_bytes_b
        ), f"K dimension mismatch between qx ({K_bytes_a}) and qw ({K_bytes_b})"
        K = 2 * K_bytes_a  # 2 FP4 values per byte

        if (
            K > 0
            and tensor_scale_x is not None
            and tensor_scale_w is not None
        ):
            factor = 6.0 * 6.0 * 448.0 * 448.0
            partial_alpha = (tensor_scale_x * tensor_scale_w).to(torch.float32)
            alpha = partial_alpha.squeeze(-1) / factor

        y = cutedsl_nvfp4_gemm_optimized(
            qx, qw, sx, sw, M, N, K,
            alpha, out_dtype,
            qx.device, out, accumulate, has_swzzile=False
        )

        return y


class NVFP4TritonBackend(GemmBackendBase):

    def gemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        grad: bool,
        accumulate: bool,
        layout: str,
        alpha: torch.Tensor,
        tensor_scale_x: torch.Tensor,
        tensor_scale_w: torch.Tensor,
        has_swzzile: bool,
    ) -> torch.Tensor:
        """
            Triton block-scaled QGEMM implementation using block_scaled_matmul.
            """
        assert bias is None, "Bias is not implemented for triton block-scaled version."

        BLOCK_M = 128
        BLOCK_N = 256
        BLOCK_K = 256
        VEC_SIZE = 16
        ELEM_PER_BYTE_A = 2
        ELEM_PER_BYTE_B = 2

        device = "cuda"
        a = qx.to(device)
        b = qw.to(device)

        M, K_bytes_a = a.shape
        N, K_bytes_b = b.shape
        K = 2 * K_bytes_a
        assert K_bytes_a == K_bytes_b, f"K dimension mismatch: {K_bytes_a} vs {K_bytes_b}"

        assert M % 128 == 0, f"M ({M}) must be divisible by 128"
        assert N % 128 == 0, f"N ({N}) must be divisible by 128"
        assert K % 64 == 0, f"K ({K}) must be divisible by 64"

        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

        # sx, sw: 4D (M//128, K//64, 32, 16) and (N//128, K//64, 32, 16) from init; use directly, no permute
        a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]  # [M//128, K//64, 32, 16]
        b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]

        # Convert 2D -> 4D for triton/cuBLAS
        if not has_swzzile:
            sx = swizzle_scale(sx, M, K).to(torch.float8_e4m3fn)
            sw = swizzle_scale(sw, N, K).to(torch.float8_e4m3fn)

        sx_reshaped = sx.to(torch.float8_e4m3fn).to(device)
        sw_reshaped = sw.to(torch.float8_e4m3fn).to(device)

        rep_m = BLOCK_M // 128
        rep_n = BLOCK_N // 128
        rep_k = BLOCK_K // VEC_SIZE // 4

        a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
        b_scale_block_shape = [1, rep_n, rep_k, 2, 256]

        sx_5d = sx_reshaped.reshape(1, sx_reshaped.shape[0], sx_reshaped.shape[1], 2, 256)
        sw_5d = sw_reshaped.reshape(1, sw_reshaped.shape[0], sw_reshaped.shape[1], 2, 256)

        sx_5d = sx_5d.to(device)
        sw_5d = sw_5d.to(device)

        a_scale_desc = TensorDescriptor.from_tensor(sx_5d, block_shape=a_scale_block_shape)
        b_scale_desc = TensorDescriptor.from_tensor(sw_5d, block_shape=b_scale_block_shape)

        configs = {
            "BLOCK_SIZE_M": BLOCK_M,
            "BLOCK_SIZE_N": BLOCK_N,
            "BLOCK_SIZE_K": BLOCK_K,
            "num_stages": 4,
            "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
            "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
            "VEC_SIZE": VEC_SIZE,
        }

        # Global alpha for NVFP4: (global_amax_qx * global_amax_qw) / (6*6*448*448). Use 1.0 if pow_2_scales or K==0.
        if K > 0 and tensor_scale_x is not None and tensor_scale_w is not None:
            factor = 6.0 * 6.0 * 448.0 * 448.0
            alpha = (tensor_scale_x * tensor_scale_w).to(torch.float32).squeeze(-1).item() / factor
        else:
            alpha = 1.0

        output_dtype_for_kernel = torch.float16
        output = block_scaled_matmul(
            a_desc, a_scale_desc, b_desc, b_scale_desc,
            alpha,
            output_dtype_for_kernel, M, N, K, rep_m, rep_n, rep_k, configs
        )

        # Handle accumulation
        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out.add_(output)
            return out
        else:
            output = output.to(out_dtype)
            return output


class NVFP4FlashInferBackend(GemmBackendBase):

    def gemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        grad: bool,
        accumulate: bool,
        layout: str,
        alpha: torch.Tensor,
        tensor_scale_x: torch.Tensor,
        tensor_scale_w: torch.Tensor,
        has_swzzile: bool,
    ) -> torch.Tensor:
        m, packed_k = qx.shape
        k = packed_k * 2
        n = qw.shape[0]
        BLOCK = 16
        y = torch.empty([m, n], device=qx.device, dtype=torch.bfloat16)

        if (
            k > 0
            and tensor_scale_x is not None
            and tensor_scale_w is not None
        ):
            factor = 6.0 * 6.0 * 448.0 * 448.0
            partial_alpha = (tensor_scale_x * tensor_scale_w).to(torch.float32)
            alpha = partial_alpha.squeeze(-1) / factor

        mm_fp4(
            qx,
            qw.T,
            sx.view(-1, k // BLOCK),
            sw.view(-1, k // BLOCK).T,
            alpha,
            torch.bfloat16,
            y,
            block_size=BLOCK,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )

        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out.add_(y)
            return out
        else:
            y = y.to(out_dtype)
            return y


def dispatch_gemm_backend(custom_gemm_backend, quantize_type) -> Union[None, GemmBackendBase]:
    if custom_gemm_backend == 'cublas':
        return NVFP4CublasGemmBackend()
    elif custom_gemm_backend == 'cutlass':
        return NVFP4CutlassGemmBackend()
    elif custom_gemm_backend == 'cuteDSL':
        return NVFP4CuteDSLGemmBackend()
    elif custom_gemm_backend == 'triton':
        return NVFP4TritonBackend()
    elif custom_gemm_backend == 'flashinfer':
        return NVFP4FlashInferBackend()

    return None