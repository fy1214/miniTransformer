"""TE fe_api-style precision tests for in-tree ``grouped_gemm_glu``.

Random FP4 + block SF → local ``grouped_gemm_glu_wrapper_sm100`` vs TE
``check_ref_grouped_gemm_swiglu`` (atol=1e-1, rtol=1e-2).

Ported from cudnn-frontend ``test/python/fe_api/test_grouped_gemm_glu.py``.
Markers: ``production``. Requires CUDA sm_100+, CuteDSL, CUTLASS.
"""

from __future__ import annotations

import pytest
import torch

from miniTransformer.ops.te_grouped_gemm_glu import is_grouped_gemm_glu_available

pytestmark = pytest.mark.production


def _import_local_wrapper():
    from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_glu import (
        grouped_gemm_glu_wrapper_sm100,
    )

    return grouped_gemm_glu_wrapper_sm100


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        return f"grouped_gemm_glu requires sm_100+ (got sm_{major}{minor})"
    if not is_grouped_gemm_glu_available():
        return "local grouped_gemm_glu not importable"
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return "torch.float4_e2m1fn_x2 unavailable"
    return None


_SKIP = _skip_reason()


# Match cudnn-frontend grouped_gemm_glu / check_ref_grouped_gemm_swiglu.
_GLU_ATOL = 1e-1
_GLU_RTOL = 1e-2

# --- FP4 pack/unpack (from cudnn-frontend test_low_precision_matmul) ---

def ceil_div(a, b):
    return (a + b - 1) // b


def _n_ones(n: int) -> int:
    return (1 << n) - 1


FP4_EBITS = 2
FP4_MBITS = 1

EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _floatx_unpacked_to_f32(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert sub-byte floating point numbers with the given number of exponent
    and mantissa bits to FP32.

    Input: torch.Tensor of dtype uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    # save the sign
    sign_lp = x & sign_mask

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    #
    # 1. Calculate zero mask
    #
    zero_mask = x_pos == 0

    #
    # 2. Calculate the denormal path mask
    #
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    #
    # 3. Calculate the normal path
    #

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    #
    # 4. Add the zero and denormal casts to the already casted normal path
    #
    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # fast path.
    # without this, performance for FP4_E2M1 is slower by 2x
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32

    else:
        # iterate over all possible values of mantissa
        # i=0, j=1
        # i=1, j=10,11
        # i=2, j=100,101,110,111
        # and so on
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                # left shift mantissa until it overflows (create an implicit 1)
                # subtract exponent by the same amount
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp - (1 << i)) << (left_shift + MBITS_F32 - mbits)
                exp_biased_f32 = (denormal_exp_biased - left_shift) << MBITS_F32

                # we can update this in-place since the values won't overlap
                # torch.compile() may complain unsupported operand type(s) for |: 'SymInt' and 'int'
                # thus we use + instead of | here
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = exp_biased_f32 + mantissa_f32

        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # add sign back
    sign_f32 = sign_lp.to(torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def unpack_uint4(packed_data):
    """Unpack uint4 data from packed uint8 format.
    Reverses the operation of pack_uint4.
    """
    shape = packed_data.shape
    # Create output shape with last dimension doubled
    unpacked_shape = (*shape[:-1], shape[-1] * 2)

    # View as uint8 and flatten
    packed_uint8 = packed_data.view(torch.uint8).contiguous().view(-1)

    # Create unpacked array
    unpacked = torch.zeros(packed_uint8.shape[0] * 2, dtype=torch.uint8, device=packed_data.device)

    # Extract lower and upper 4 bits
    unpacked[::2] = packed_uint8 & 0x0F  # Lower 4 bits
    unpacked[1::2] = (packed_uint8 >> 4) & 0x0F  # Upper 4 bits

    return unpacked.view(unpacked_shape)


def _bfloat16_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


def float4_e2m1fn_x2_to_float32(fp4_tensor):
    """Convert torch.float4_e2m1fn_x2 tensor to torch.float32."""
    # View as uint8 and unpack
    unpacked_uint8 = unpack_uint4(fp4_tensor)

    # Convert to float32 using the existing conversion function
    f32_tensor = _floatx_unpacked_to_f32(unpacked_uint8, FP4_EBITS, FP4_MBITS)

    return f32_tensor



try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack as _cute_from_dlpack

    @cute.jit
    def _cvt_sf_MKL_to_M32x4xrm_K4xrk_L(sf_ref_tensor: cute.Tensor, sf_mma_tensor: cute.Tensor):
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
        for i in cutlass.range(cute.size(sf_ref_tensor)):
            mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
            sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

except Exception:  # noqa: BLE001
    cute = None
    _cute_from_dlpack = None
    _cvt_sf_MKL_to_M32x4xrm_K4xrk_L = None


def _cvt_sf_mkl_to_mma(sf_ref_cpu: torch.Tensor, sf_mma_cpu: torch.Tensor) -> None:
    if _cvt_sf_MKL_to_M32x4xrm_K4xrk_L is None or _cute_from_dlpack is None:
        pytest.skip("CUTLASS required for TE-style SF layout")
    _cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        _cute_from_dlpack(sf_ref_cpu),
        _cute_from_dlpack(sf_mma_cpu),
    )


def _create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    min_val, max_val = -2, 2
    if dtype not in {torch.float4_e2m1fn_x2, torch.uint8}:
        dtype_tensor = (
            torch.empty(shape, dtype=torch.float32, device="cuda")
            .uniform_(float(min_val), float(max_val))
            .permute(permute_order)
            .to(dtype)
        )
        ref_tensor = dtype_tensor.to(torch.float32)
    else:
        dtype_tensor = _bfloat16_to_float4_e2m1fn_x2(
            torch.empty(shape, dtype=torch.float32, device="cuda")
            .uniform_(float(min_val), float(max_val))
            .to(torch.bfloat16)
        )
        ref_tensor = float4_e2m1fn_x2_to_float32(dtype_tensor).to(torch.float32).permute(permute_order)
        dtype_tensor = dtype_tensor.permute(permute_order).view(dtype)
    return ref_tensor, dtype_tensor


def _create_sf_layout_tensor(l, mn, nk, sf_vec_size):
    sf_k = ceil_div(nk, sf_vec_size)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    cute_f32 = torch.zeros(mma_shape, dtype=torch.float32).permute((3, 4, 1, 5, 2, 0))
    return cute_f32, sf_k


def _create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
    cute_f32, sf_k = _create_sf_layout_tensor(l, mn, k, sf_vec_size)
    ref = (
        torch.empty((l, mn, sf_k), dtype=torch.float32)
        .uniform_(1, 3)
        .permute((1, 2, 0))
        .to(torch.int8)
        .to(torch.float32)
    )
    if ref.numel() > 0:
        _cvt_sf_mkl_to_mma(ref, cute_f32)
    ref = (
        ref.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(1, 2, 0)
    )
    ref = ref[:, :k, :]
    cute = cute_f32.to(dtype).cuda()
    return ref.cuda(), cute


def _te_create_mask(group_m_list, m_aligned=256):
    valid_m = 0
    aligned = []
    offsets = []
    for group_m in group_m_list:
        am = ((group_m + m_aligned - 1) // m_aligned) * m_aligned
        valid_m += am
        aligned.append(am)
        offsets.append(valid_m)
    return valid_m, aligned, torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _te_allocate_inputs(n, k, l, group_m_list, sf_vec_size, sf_dtype, m_aligned=256):
    valid_m, aligned, padded_offsets = _te_create_mask(group_m_list, m_aligned)
    a_ref, a_tensor = _create_and_permute_tensor(1, valid_m, k, False, torch.float4_e2m1fn_x2)
    b_ref, b_tensor = _create_and_permute_tensor(l, n, k, False, torch.float4_e2m1fn_x2)
    sfa_ref, sfa_tensor = _create_scale_factor_tensor(1, valid_m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor = _create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)
    alpha = torch.randint(-2, 2, (l,), dtype=torch.float32, device="cuda").float()
    prob = torch.randint(-2, 2, (valid_m, 1, 1), dtype=torch.float32, device="cuda").float()
    return {
        "a_tensor": a_tensor,
        "a_ref": a_ref,
        "b_tensor": b_tensor,
        "b_ref": b_ref,
        "sfa_tensor": sfa_tensor,
        "sfa_ref": sfa_ref,
        "sfb_tensor": sfb_tensor,
        "sfb_ref": sfb_ref,
        "alpha_tensor": alpha,
        "prob_tensor": prob,
        "padded_offsets_tensor": padded_offsets,
        "aligned_group_m_list": aligned,
        "valid_m": valid_m,
    }


def _te_run_ref_swiglu(
    a_ref,
    b_ref,
    sfa_ref,
    sfb_ref,
    alpha_tensor,
    prob_tensor,
    aligned_group_m_list,
    valid_m,
    *,
    per_token_scale: bool = False,
    alpha_b_tensor=None,
    d_dtype=torch.bfloat16,
):
    """TE ``run_grouped_gemm_swiglu_ref`` (+ optional bilateral ``alpha_b``)."""
    n, k, l = b_ref.shape
    n_out = n // 2
    ref = torch.empty((1, valid_m, n), dtype=torch.float32, device=a_ref.device)
    start = 0
    for i, group_m in enumerate(aligned_group_m_list):
        end = start + group_m
        res_a = torch.einsum("mk,mk->mk", a_ref[start:end, :, 0], sfa_ref[start:end, :, 0])
        res_b = torch.einsum("nk,nk->nk", b_ref[:, :, i], sfb_ref[:, :, i])
        ref[0, start:end, :] = torch.einsum("mk,nk->mn", res_a, res_b)
        start = end
    ref = ref.permute((1, 2, 0))

    if per_token_scale:
        ref = ref * alpha_tensor.to(torch.float32).reshape(valid_m, 1, 1)
    else:
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            ref[start:end, :, 0] = ref[start:end, :, 0] * float(alpha_tensor[i])
            start = end

    if alpha_b_tensor is not None:
        # α_b: (N, E) stride (1, N) — ACC *= α_a[m] * α_b[n]
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            ref[start:end, :, 0] = ref[start:end, :, 0] * alpha_b_tensor[:, i].float().unsqueeze(0)
            start = end

    c_ref = ref.clone()

    group = 32
    cols = torch.arange(n, device=ref.device, dtype=torch.long)
    block_cols = cols.view(n // group, group)
    gate_idx = block_cols[0::2].reshape(-1)
    up_idx = block_cols[1::2].reshape(-1)
    ref_gate = ref.index_select(1, gate_idx)
    ref_up = ref.index_select(1, up_idx)
    ref_gate = ref_gate * torch.sigmoid(ref_gate)
    d_ref = ref_up * ref_gate
    d_ref = d_ref * prob_tensor.expand(-1, n_out, -1)

    if per_token_scale:
        amax_ref = d_ref.abs().amax(dim=(1, 2))
    else:
        amax_ref = torch.empty((l,), dtype=torch.float32, device=a_ref.device)
        start = 0
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            amax_ref[i] = d_ref[start:end, :, 0].abs().amax()
            start = end
    return {"c_ref": c_ref, "d_ref": d_ref, "amax_ref": amax_ref}


def _te_check_ref(
    inputs,
    outputs,
    *,
    per_token_scale: bool = False,
    d_dtype=torch.bfloat16,
    c_dtype=torch.bfloat16,
    atol=_GLU_ATOL,
    rtol=_GLU_RTOL,
    check_d: bool = True,
):
    """TE ``check_ref_grouped_gemm_swiglu`` (bf16/fp32 D path)."""
    b_ref = inputs.get("b_ref")
    sfb_ref = inputs.get("sfb_ref")
    if b_ref is None:
        # Discrete: stack per-expert refs to (n, k, l)
        b_ref = torch.cat(
            [b.to(torch.float32) for b in inputs["b_ref_list"]], dim=-1
        )
        sfb_ref = torch.cat(
            [s.to(torch.float32) for s in inputs["sfb_ref_list"]], dim=-1
        )
    else:
        b_ref = b_ref.to(torch.float32)
        sfb_ref = sfb_ref.to(torch.float32)

    refs = _te_run_ref_swiglu(
        a_ref=inputs["a_ref"].to(torch.float32),
        b_ref=b_ref,
        sfa_ref=inputs["sfa_ref"].to(torch.float32),
        sfb_ref=sfb_ref,
        alpha_tensor=inputs["alpha_tensor"],
        prob_tensor=inputs["prob_tensor"],
        aligned_group_m_list=inputs["aligned_group_m_list"],
        valid_m=inputs["valid_m"],
        per_token_scale=per_token_scale,
        alpha_b_tensor=inputs.get("alpha_b_tensor"),
        d_dtype=d_dtype,
    )
    torch.cuda.synchronize()
    valid_m = inputs["valid_m"]
    torch.testing.assert_close(
        outputs["c_tensor"][:valid_m].cpu().float(),
        refs["c_ref"].cpu().to(c_dtype).to(torch.float32),
        atol=atol,
        rtol=rtol,
    )
    if not check_d:
        return
    torch.testing.assert_close(
        outputs["d_tensor"][:valid_m].cpu().float(),
        refs["d_ref"].cpu().to(d_dtype).to(torch.float32),
        atol=atol,
        rtol=rtol,
    )
    if outputs.get("amax_tensor") is not None:
        torch.testing.assert_close(
            outputs["amax_tensor"].cpu().squeeze(),
            refs["amax_ref"].cpu(),
            atol=atol,
            rtol=rtol,
        )


def _te_allocate_discrete_inputs(
    n, k, num_experts, group_m_list, sf_vec_size, sf_dtype, m_aligned=256, b_major="k"
):
    """TE discrete allocate: per-expert B/SFB + int64 pointer tensors."""
    valid_m, aligned, padded_offsets = _te_create_mask(group_m_list, m_aligned)
    a_ref, a_tensor = _create_and_permute_tensor(1, valid_m, k, False, torch.float4_e2m1fn_x2)
    b_list, b_ref_list, sfb_list, sfb_ref_list = [], [], [], []
    for _ in range(num_experts):
        b_ref_i, b_tensor_i = _create_and_permute_tensor(1, n, k, b_major == "n", torch.float4_e2m1fn_x2)
        b_list.append(b_tensor_i.squeeze(-1) if b_tensor_i.shape[-1] == 1 else b_tensor_i)
        b_ref_list.append(b_ref_i)
        sfb_ref_i, sfb_tensor_i = _create_scale_factor_tensor(1, n, k, sf_vec_size, sf_dtype)
        sfb_list.append(sfb_tensor_i)
        sfb_ref_list.append(sfb_ref_i)
    sfa_ref, sfa_tensor = _create_scale_factor_tensor(1, valid_m, k, sf_vec_size, sf_dtype)
    alpha = torch.randint(-2, 2, (num_experts,), dtype=torch.float32, device="cuda").float()
    prob = torch.randint(-2, 2, (valid_m, 1, 1), dtype=torch.float32, device="cuda").float()
    b_ptrs = torch.tensor([b.data_ptr() for b in b_list], dtype=torch.int64, device="cuda")
    sfb_ptrs = torch.tensor([s.data_ptr() for s in sfb_list], dtype=torch.int64, device="cuda")
    return {
        "a_tensor": a_tensor,
        "a_ref": a_ref,
        "b_list": b_list,
        "b_ref_list": b_ref_list,
        "b_ptrs_tensor": b_ptrs,
        "sfa_tensor": sfa_tensor,
        "sfa_ref": sfa_ref,
        "sfb_list": sfb_list,
        "sfb_ref_list": sfb_ref_list,
        "sfb_ptrs_tensor": sfb_ptrs,
        "alpha_tensor": alpha,
        "prob_tensor": prob,
        "padded_offsets_tensor": padded_offsets,
        "aligned_group_m_list": aligned,
        "valid_m": valid_m,
    }


def _make_alpha_b(n: int, l: int) -> torch.Tensor:
    """(N, E) fp32 with stride (1, N) — same layout as production ``build_per_n_alpha``."""
    cols = [
        (0.5 + 0.1 * torch.arange(n, device="cuda", dtype=torch.float32) + e)
        for e in range(l)
    ]
    # (E, N) contiguous → .t() → (N, E) stride (1, N); do not re-contiguous.
    return torch.stack(cols, dim=0).t()


def _launch_dense(inputs, *, mma_tiler_mn, cluster_shape_mn, sf_vec_size, vector_f32,
                  use_dynamic_sched, per_token_scale, d_dtype=torch.bfloat16,
                  alpha_b_tensor=None):
    from cuda.bindings import driver as cuda  # type: ignore

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    wrapper = _import_local_wrapper()
    kwargs = dict(
        a_tensor=inputs["a_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        b_tensor=inputs["b_tensor"],
        sfb_tensor=inputs["sfb_tensor"],
        prob_tensor=inputs["prob_tensor"],
        acc_dtype=torch.float32,
        c_dtype=torch.bfloat16,
        d_dtype=d_dtype,
        cd_major="n",
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        m_aligned=256,
        discrete_col_sfd=False,
        act_func="swiglu",
        use_dynamic_sched=use_dynamic_sched,
        per_token_scale=per_token_scale,
        current_stream=stream,
    )
    if alpha_b_tensor is not None:
        kwargs["alpha_b_tensor"] = alpha_b_tensor
    return wrapper(**kwargs)


def _dense_fp4_legal_cases():
    """FP4 recipes allowed by ``GroupedGemmGluSm100.check_support``.

    Excludes illegal combos (do not collect / skip them):
    - ``sf_vec=32`` + ``e4m3`` (MXFP4 needs e8m0)
    - ``sf_vec=16`` + ``d_dtype=float32`` (NVFP4 needs bf16 D)
    - ``mma_tiler_m=256`` (2CTA) + ``cluster_shape_mn[0]=1`` (must be divisible by 2)
    """
    cases = []
    for use_dynamic_sched in (False, True):
        for vector_f32 in (True, False):
            for mma_tiler_mn in ((256, 256), (128, 256)):
                # 2CTA (M=256) requires cluster M divisible by 2; M=128 allows 1x1 or 2x1.
                clusters = ((2, 1),) if mma_tiler_mn[0] == 256 else ((2, 1), (1, 1))
                for cluster_shape_mn in clusters:
                    # NVFP4: sf16 + e8m0/e4m3, D=bf16 only
                    for sf_vec_size, sf_dtype, sf_id in (
                        (16, torch.float8_e8m0fnu, "sf16_e8m0"),
                        (16, torch.float8_e4m3fn, "sf16_e4m3"),
                    ):
                        cases.append(
                            pytest.param(
                                use_dynamic_sched,
                                torch.bfloat16,
                                cluster_shape_mn,
                                vector_f32,
                                mma_tiler_mn,
                                sf_vec_size,
                                sf_dtype,
                                id=(
                                    f"{sf_id}-mma{mma_tiler_mn[0]}x{mma_tiler_mn[1]}-"
                                    f"{'vf32' if vector_f32 else 'nof32'}-"
                                    f"c{cluster_shape_mn[0]}x{cluster_shape_mn[1]}-"
                                    f"bf16-{'dyn' if use_dynamic_sched else 'static'}"
                                ),
                            )
                        )
                    # MXFP4: sf32 + e8m0 only; D=bf16 or fp32
                    for d_dtype, d_id in (
                        (torch.bfloat16, "bf16"),
                        (torch.float32, "fp32"),
                    ):
                        cases.append(
                            pytest.param(
                                use_dynamic_sched,
                                d_dtype,
                                cluster_shape_mn,
                                vector_f32,
                                mma_tiler_mn,
                                32,
                                torch.float8_e8m0fnu,
                                id=(
                                    f"sf32_e8m0-mma{mma_tiler_mn[0]}x{mma_tiler_mn[1]}-"
                                    f"{'vf32' if vector_f32 else 'nof32'}-"
                                    f"c{cluster_shape_mn[0]}x{cluster_shape_mn[1]}-"
                                    f"{d_id}-{'dyn' if use_dynamic_sched else 'static'}"
                                ),
                            )
                        )
    return cases


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="torch.float4_e2m1fn_x2 unavailable",
)
class TestGroupedGemmGluTeStyleCheckRef:
    """TE fe_api-style checks: dense FP4 (legal recipes), per-token α, discrete."""

    # ---- dense FP4: only check_support-legal (sf, D) recipes ----
    @pytest.mark.parametrize(
        "use_dynamic_sched,d_dtype,cluster_shape_mn,vector_f32,mma_tiler_mn,sf_vec_size,sf_dtype",
        _dense_fp4_legal_cases(),
    )
    def test_dense_wrapper_fp4(
        self,
        use_dynamic_sched,
        d_dtype,
        cluster_shape_mn,
        vector_f32,
        mma_tiler_mn,
        sf_vec_size,
        sf_dtype,
    ):
        torch.manual_seed(0)
        n, k, l = 256, 256, 2
        inputs = _te_allocate_inputs(n, k, l, [256] * l, sf_vec_size, sf_dtype)
        outputs = _launch_dense(
            inputs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            use_dynamic_sched=use_dynamic_sched,
            per_token_scale=False,
            d_dtype=d_dtype,
        )
        _te_check_ref(inputs, outputs, per_token_scale=False, d_dtype=d_dtype)

    # ---- per-token: α_a only ----
    def test_dense_wrapper_per_token_alpha_a_only(self):
        torch.manual_seed(0)
        n, k, l = 256, 256, 2
        inputs = _te_allocate_inputs(n, k, l, [256] * l, 16, torch.float8_e4m3fn)
        valid_m = inputs["valid_m"]
        token_idx = torch.arange(valid_m, device="cuda", dtype=torch.float32)
        inputs["alpha_tensor"] = (0.25 + (token_idx % 17) * 0.05).reshape(valid_m, 1, 1)
        outputs = _launch_dense(
            inputs,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            sf_vec_size=16,
            vector_f32=False,
            use_dynamic_sched=True,
            per_token_scale=True,
        )
        assert tuple(outputs["amax_tensor"].shape) == (valid_m, 1, 1)
        _te_check_ref(inputs, outputs, per_token_scale=True)

    # ---- per-token: α_a × α_b ----
    def test_dense_wrapper_per_token_alpha_ab(self):
        torch.manual_seed(0)
        n, k, l = 256, 256, 2
        inputs = _te_allocate_inputs(n, k, l, [256] * l, 16, torch.float8_e4m3fn)
        valid_m = inputs["valid_m"]
        token_idx = torch.arange(valid_m, device="cuda", dtype=torch.float32)
        inputs["alpha_tensor"] = (0.25 + (token_idx % 17) * 0.05).reshape(valid_m, 1, 1)
        alpha_b = _make_alpha_b(n, l)
        inputs["alpha_b_tensor"] = alpha_b
        outputs = _launch_dense(
            inputs,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            sf_vec_size=16,
            vector_f32=False,
            use_dynamic_sched=True,
            per_token_scale=True,
            alpha_b_tensor=alpha_b,
        )
        assert tuple(outputs["amax_tensor"].shape) == (valid_m, 1, 1)
        _te_check_ref(inputs, outputs, per_token_scale=True)

    # ---- discrete FP4 (TE discrete_wrapper_fp4: swiglu/geglu × sched) ----
    @pytest.mark.parametrize("use_dynamic_sched", [False, True], ids=["static", "dynamic"])
    @pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
    def test_discrete_wrapper_fp4(self, use_dynamic_sched, act_func):
        from cuda.bindings import driver as cuda  # type: ignore

        torch.manual_seed(0)
        n, k, l = 256, 256, 2
        # TE discrete fp4 default: sf_vec=32, e8m0, mma 256, cluster 2x1
        inputs = _te_allocate_discrete_inputs(
            n, k, l, [256] * l, 32, torch.float8_e8m0fnu, b_major="k"
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        wrapper = _import_local_wrapper()
        outputs = wrapper(
            a_tensor=inputs["a_tensor"],
            sfa_tensor=inputs["sfa_tensor"],
            padded_offsets=inputs["padded_offsets_tensor"],
            alpha_tensor=inputs["alpha_tensor"],
            b_ptrs=inputs["b_ptrs_tensor"],
            sfb_ptrs=inputs["sfb_ptrs_tensor"],
            n=n,
            b_dtype=inputs["b_list"][0].dtype,
            b_major="k",
            prob_tensor=inputs["prob_tensor"],
            acc_dtype=torch.float32,
            c_dtype=torch.bfloat16,
            d_dtype=torch.bfloat16,
            cd_major="n",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            sf_vec_size=32,
            vector_f32=False,
            m_aligned=256,
            discrete_col_sfd=False,
            act_func=act_func,
            use_dynamic_sched=use_dynamic_sched,
            per_token_scale=False,
            current_stream=stream,
        )
        # TE: GeGLU ref only checks C (activation math differs from SwiGLU golden).
        _te_check_ref(
            inputs,
            outputs,
            per_token_scale=False,
            check_d=(act_func == "swiglu"),
        )
