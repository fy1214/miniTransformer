// Split-amax grouped per-token NVFP4 quantization.
//
// Two kernels:
//   1. fp4_perrow_amax_te_kernel — TE-style per-row absolute-max reduction.
//      (grid=num_rows, block=256, universal — no Blackwell dependency)
//   2. fp4_group_pertoken_split_amax_kernel — pure streaming quantize with
//      pre-computed amax. (flat 1-D grid, zero __syncthreads, zero shared
//      memory, no K/M limits)
//
// Together they implement the split_amax=True path in
// group_quantize_nvfp4_pertoken_swizzled().

#include <cuda_bf16.h>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#define CUDA_CHECK(expr) TORCH_CHECK((expr) == cudaSuccess, "CUDA error: " #expr)

// ── Per-row absolute max (TE-style, grid=num_rows, block=256) ─────────────────
// Ported from TransformerEngine common/cast/nvfp4/quantize_transpose_nvfp4.cuh
// compute_rowwise_amax_kernel.
// Uses bf16x2 vectorized loads, half-precision __hmax/__habs, block reduce.
// Only 1 __syncthreads (inside reduce_max).


// Ported from TransformerEngine common/cast/nvfp4/quantize_transpose_nvfp4.cuh
// compute_rowwise_amax_kernel
//
// Key design: grid=num_rows, block=256, each block fully reduces one row.
// Uses bf16x2 vectorized loads, half-precision __hmax/__habs, block reduce.
// Only 1 __syncthreads (inside reduce_max).

static constexpr int TE_AMAX_BLOCK = 256;
static constexpr int TE_AMAX_WARPS = TE_AMAX_BLOCK / 32;  // 8

template<int num_elems>
__device__ __forceinline__ float te_warp_reduce_max(float m) {
#pragma unroll
    for (int delta = num_elems / 2; delta > 0; delta /= 2) {
        float other = __shfl_down_sync(0xffffffff, m, delta);
        m = fmaxf(m, other);
    }
    return m;
}

template<int num_warps>
__device__ __forceinline__ float te_block_reduce_max(float m, int warpid) {
    __shared__ float staging[num_warps];
    float warp_max = te_warp_reduce_max<32>(m);
    if (threadIdx.x % 32 == 0)
        staging[warpid] = warp_max;
    __syncthreads();
    float result = 0.f;
    if (warpid == 0) {
        float val = threadIdx.x < num_warps ? staging[threadIdx.x] : 0.f;
        result = te_warp_reduce_max<num_warps>(val);
    }
    return result;
}

__global__ void __launch_bounds__(TE_AMAX_BLOCK)
fp4_perrow_amax_te_kernel(
    float*              __restrict__ amax_out,
    const nv_bfloat16*  __restrict__ x_ptr,
    int num_rows,
    int num_cols)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const int num_vec2 = num_cols / 2;
    const nv_bfloat162* input_row = reinterpret_cast<const nv_bfloat162*>(x_ptr + row_idx * num_cols);
    const int warp_id = threadIdx.x / 32;

    nv_bfloat16 max_val = static_cast<nv_bfloat16>(0.0f);
    for (int i = threadIdx.x; i < num_vec2; i += TE_AMAX_BLOCK) {
        nv_bfloat162 v = input_row[i];
        max_val = __hmax(__habs(v.x), max_val);
        max_val = __hmax(__habs(v.y), max_val);
    }

    float thread_max = static_cast<float>(max_val);
    float row_max = te_block_reduce_max<TE_AMAX_WARPS>(thread_max, warp_id);

    if (threadIdx.x == 0)
        amax_out[row_idx] = row_max;
}

static void launch_fp4_perrow_amax_te(
    float*             amax_out,
    const nv_bfloat16* x_ptr,
    long   total_rows,
    long   cols,
    cudaStream_t stream)
{
    dim3 grid(static_cast<int>(total_rows));
    dim3 block(TE_AMAX_BLOCK);
    fp4_perrow_amax_te_kernel<<<grid, block, 0, stream>>>(
        amax_out, x_ptr,
        static_cast<int>(total_rows), static_cast<int>(cols));
    CUDA_CHECK(cudaGetLastError());
}

void fp4_perrow_amax_te_binding(
    at::Tensor&       amax_out,
    const at::Tensor& inp)
{
    TORCH_CHECK(inp.dtype()   == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                  "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),            "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                 "inp must be 2D [total_m, cols]");
    TORCH_CHECK(inp.size(1) % 2 == 0,           "cols must be even");
    TORCH_CHECK(amax_out.numel() == inp.size(0), "amax_out size mismatch");
    TORCH_CHECK(amax_out.dtype() == at::kFloat,  "amax_out must be float32");

    launch_fp4_perrow_amax_te(
        reinterpret_cast<float*>(amax_out.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        inp.size(0), inp.size(1),
        at::cuda::getCurrentCUDAStream());
}


// ── split_amax streaming quantize kernel (Blackwell-only) ────────────────────────────

#ifndef MINI_TE_NO_BLACKWELL_OPS

#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "../../../util/quarter/utils.cuh"
#include "../../../util/quarter/vec.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp4x8  = GenericVector<unsigned char, 4>;

struct QuantResultGP_SA {
    fp4x8 bits;
    __nv_fp8_e4m3 fp8s;
};

__device__ __forceinline__ nv_bfloat16 vecReduceAbsMax_SA(bf16x8 v) {
    nv_bfloat16 m = __habs(v[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) m = __hmax(m, __habs(v[i]));
    return m;
}

__device__ __forceinline__ QuantResultGP_SA quantize_gp_SA(
    float group_abs_max, float fp4_max, float global_encode_scale,
    float token_decode_scale, bf16x8 x)
{
    float local_sf = (group_abs_max == 0.f) ? 1.f : group_abs_max / fp4_max;
    float enc_sf = local_sf * global_encode_scale;
    __nv_fp8_e4m3 fp8s = __nv_fp8_e4m3(enc_sf);
    float descale = static_cast<float>(fp8s) * token_decode_scale;
    float inv = (descale == 0.f) ? 0.f : 1.0f / descale;
    QuantResultGP_SA res;
    res.fp8s = fp8s;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 pair = {static_cast<float>(x[2*j]) * inv,
                       static_cast<float>(x[2*j+1]) * inv};
        res.bits[j] = __nv_cvt_float2_to_fp4x2(pair, __NV_SATFINITE, __NV_E2M1);
    }
    return res;
}


//

template<int NumCandidates>
__global__ void fp4_group_pertoken_split_amax_kernel(
    __nv_fp4x4_e2m1*    __restrict__ y_ptr,
    __nv_fp8_e4m3*      __restrict__ scale_ptr,
    float*              __restrict__ global_scale_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const float*        __restrict__ amax_per_row,
    const int32_t*      __restrict__ row_offsets,
    const int64_t*      __restrict__ scale_offsets,
    float  scale_override,
    int    nvecs_per_row,
    int    cols_div16,
    int    num_groups,
    int    total_vecs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vecs) return;

    const int row        = idx / nvecs_per_row;
    const int col_in_row = idx % nvecs_per_row;

    // O(log G) binary search for owning group.
    int group_id = 0;
    {
        int lo = 0, hi = num_groups - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (row_offsets[mid] <= row) lo = mid;
            else                          hi = mid - 1;
        }
        group_id = lo;
    }
    const int     local_row_in_group = row - row_offsets[group_id];
    const int64_t sf_base            = scale_offsets[group_id];

    // Load bf16x8.
    bf16x8 x = bf16x8::load(x_ptr + 8 * idx);

    // Token-level global scale from pre-computed per-row amax.
    float row_abs_max = amax_per_row[row];
    constexpr float SCALES_MAX = NumCandidates > 1 ? 256.0f : 448.0f;
    constexpr float FP4_MAX    = 6.0f;
    float global_encode_scale = row_abs_max == 0.f
                ? 1.f
                : (SCALES_MAX * FP4_MAX) / row_abs_max;
    float scale = row_abs_max == 0.f
                ? scale_override
                : scale_override / global_encode_scale;

    if (col_in_row == 0)
        global_scale_ptr[row] = scale;

    // Per-group abs-max: 16 elements = 2 consecutive threads.
    nv_bfloat16 local_abs_max = vecReduceAbsMax_SA(x);
    nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
    float full_abs_max = static_cast<float>(
        __hmax(__habs(local_abs_max), __habs(other_abs_max)));

    // Quantize.
    QuantResultGP_SA res;
    if constexpr (NumCandidates == 1) {
        res = quantize_gp_SA(full_abs_max, 6.0f, global_encode_scale, scale, x);
    } else {
        float candidates[2] = {6.f, 4.f};
        QuantResultGP_SA results[2];
        float best = __builtin_inff();
        for (int i = 0; i < 2; ++i) {
            results[i] = quantize_gp_SA(full_abs_max, candidates[i],
                                     global_encode_scale, scale, x);
            const float descale = static_cast<float>(results[i].fp8s) * scale;
            float2 sum = {0.f, 0.f};
            const float2 dsv = {-descale, -descale};
            for (int j = 0; j < 4; ++j) {
                float2 dq = __nv_cvt_fp4x2_to_float2(results[i].bits[j]);
                float2 xv = {static_cast<float>(x[2*j]),
                             static_cast<float>(x[2*j+1])};
                float2 d  = __ffma2_rn(dq, dsv, xv);
                sum = __ffma2_rn(d, d, sum);
            }
            float local_err = sum.x + sum.y;
            local_err += __shfl_xor_sync(0xffffffff, local_err, 1);
            if (local_err < best) { best = local_err; res = results[i]; }
        }
    }

    // Store packed FP4.
    res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * idx);

    // Store per-group FP8 scale with per-group swizzle.
    if (col_in_row % 2 == 0) {
        const int     col       = col_in_row / 2;
        const int     numKTiles = cols_div16 / 4;
        const int64_t tgt = cvt_quant_to_fp4_get_sf_out_offset(
            local_row_in_group, col, numKTiles);
        scale_ptr[sf_base + tgt] = res.fp8s;
    }
}

// ---- Launch helper (v2: flat grid, no smem) ----

static void launch_fp4_group_pertoken_split_amax(
    int num_candidates,
    __nv_fp4x4_e2m1*   y_ptr,
    __nv_fp8_e4m3*     scale_ptr,
    float*             global_scale_ptr,
    const nv_bfloat16* x_ptr,
    const float*       amax_per_row,
    const int32_t*     row_offsets,
    const int64_t*     scale_offsets,
    float  scale_override,
    long   total_rows,
    long   cols,
    int    num_groups,
    cudaStream_t stream)
{
    TORCH_CHECK(cols % 128 == 0,
                "split_amax streaming quantize: cols must be divisible by 128, got ", cols);

    const int nvecs_per_row = static_cast<int>(cols / 8);
    const int total_vecs    = static_cast<int>(total_rows) * nvecs_per_row;
    const int cdiv16        = static_cast<int>(cols / 16);
    constexpr int BLOCK = 256;
    const int n_blocks = (total_vecs + BLOCK - 1) / BLOCK;

    if (num_candidates == 2)
        fp4_group_pertoken_split_amax_kernel<2><<<n_blocks, BLOCK, 0, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row,
            row_offsets, scale_offsets, scale_override,
            nvecs_per_row, cdiv16, num_groups, total_vecs);
    else
        fp4_group_pertoken_split_amax_kernel<1><<<n_blocks, BLOCK, 0, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr, amax_per_row,
            row_offsets, scale_offsets, scale_override,
            nvecs_per_row, cdiv16, num_groups, total_vecs);

    CUDA_CHECK(cudaGetLastError());
}

// ---- PyTorch bindings (v2) ----

void rtn_fp4_group_pertoken_split_amax_binding(
    at::Tensor&       out,
    at::Tensor&       scales,
    at::Tensor&       global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()   == at::kBFloat16,  "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                   "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),             "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                  "inp must be 2D [total_m, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,          "cols must be divisible by 128");
    TORCH_CHECK(amax_per_row.numel() == inp.size(0), "amax_per_row size mismatch");
    TORCH_CHECK(amax_per_row.dtype() == at::kFloat,  "amax_per_row must be float32");
    TORCH_CHECK(global_scale.numel() == inp.size(0), "global_scale size mismatch");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt,   "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong,  "scale_offsets must be int64");

    const long total_m = inp.size(0);
    const long cols    = inp.size(1);
    const int  ngroups = static_cast<int>(row_offsets.numel()) - 1;

    launch_fp4_group_pertoken_split_amax(
        1,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const float*>(amax_per_row.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        scale_override, total_m, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}

void four_six_fp4_group_pertoken_split_amax_binding(
    at::Tensor&       out,
    at::Tensor&       scales,
    at::Tensor&       global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()   == at::kBFloat16,  "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                   "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),             "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                  "inp must be 2D [total_m, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,          "cols must be divisible by 128");
    TORCH_CHECK(amax_per_row.numel() == inp.size(0), "amax_per_row size mismatch");
    TORCH_CHECK(amax_per_row.dtype() == at::kFloat,  "amax_per_row must be float32");
    TORCH_CHECK(global_scale.numel() == inp.size(0), "global_scale size mismatch");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt,   "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong,  "scale_offsets must be int64");

    const long total_m = inp.size(0);
    const long cols    = inp.size(1);
    const int  ngroups = static_cast<int>(row_offsets.numel()) - 1;

    launch_fp4_group_pertoken_split_amax(
        2,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const float*>(amax_per_row.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        scale_override, total_m, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}



#else  // MINI_TE_NO_BLACKWELL_OPS — stubs

#include "../../../ops.h"
#include <stdexcept>


void rtn_fp4_group_pertoken_split_amax_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override) {
  throw std::runtime_error("rtn_fp4_group_pertoken_split_amax_binding requires Blackwell+ (SM100).");
}

void four_six_fp4_group_pertoken_split_amax_binding(
    at::Tensor& out,
    at::Tensor& scales,
    at::Tensor& global_scale,
    const at::Tensor& inp,
    const at::Tensor& amax_per_row,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float scale_override) {
  throw std::runtime_error("four_six_fp4_group_pertoken_split_amax_binding requires Blackwell+ (SM100).");
}


#endif  // MINI_TE_NO_BLACKWELL_OPS
