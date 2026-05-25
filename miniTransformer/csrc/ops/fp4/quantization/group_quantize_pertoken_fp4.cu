// Per-token NV FP4 grouped quantization with per-group tcgen05 swizzled SF layout.
//
// Extends fp4_pertoken_kernel to handle multiple row-groups in one kernel launch.
// Each group's scale factor is independently swizzled (local row index within the group),
// so the output is directly usable by the CuTeDSL grouped GEMM without any CPU layout
// conversion (eliminating the D2H → CPU CuTeDSL kernel → H2D path).
//
// Grid  : (total_m,)  — one block per row, same as the single-group kernel.
// Block : (cols/8,)   — same constraint: cols divisible by 128, cols <= 4096.
//
// Scale buffer layout:
//   scale_ptr is a flat buffer partitioned by group.
//   scale_offsets[g] gives the start (in FP8 elements) of group g's scale data.
//   Each group g occupies  ceil(m_g/128)*128 * (cols/16)  elements (zero-padded to
//   the nearest 128-row tile boundary so swizzle addresses never escape the buffer).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../../../util/quarter/utils.cuh"
#include "../../../util/quarter/vec.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;
using fp4x8  = GenericVector<unsigned char, 4>;

struct QuantResultGP {
    fp4x8        bits;
    float        scale;
    __nv_fp8_e4m3 fp8s;
};

__device__ __forceinline__ QuantResultGP
quantize_gp(float abs_max, float inv_val_max, float scale, bf16x8& x)
{
    float s_group      = abs_max * inv_val_max;
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
    float s_round_fp8  = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0.f) s_round_fp8 = 1.f;

    float factor   = 1.0f / (s_round_fp8 * scale);
    float2 factor2 = {factor, factor};
    fp4x8 result;
    for (int k = 0; k < bf16x8::size; k += 2) {
        float2 src  = make_float2(static_cast<float>(x[k]), static_cast<float>(x[k + 1]));
        float2 sc   = __fmul2_rn(src, factor2);
        result[k/2] = __nv_cvt_float2_to_fp4x2(
            sc, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
    }
    return QuantResultGP{result, s_round_fp8, s_as_fp8};
}

// NumCandidates=1 : RTN  (FP4 max value = 6)
// NumCandidates=2 : Round-4-or-6  (picks lower L2-error candidate)
template<int NumCandidates>
__global__ void fp4_group_pertoken_kernel(
    __nv_fp4x4_e2m1*    __restrict__ y_ptr,
    __nv_fp8_e4m3*      __restrict__ scale_ptr,
    float*              __restrict__ global_scale_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ row_offsets,    // [num_groups+1]
    const int64_t*      __restrict__ scale_offsets,  // [num_groups+1]
    float  scale_override,
    int    cols_div16,   // cols / 16
    int    num_groups)
{
    extern __shared__ nv_bfloat16 smem[];

    const int row          = blockIdx.x;
    const int tid          = threadIdx.x;
    const int nvecs_per_row = blockDim.x;   // cols / 8

    // ---- Find which group owns this row (linear scan, cheap for ~64 groups) ----
    int group_id = 0;
    while (group_id < num_groups - 1 && row >= row_offsets[group_id + 1]) {
        ++group_id;
    }
    const int    local_row = row - row_offsets[group_id];
    const int64_t sf_base  = scale_offsets[group_id];

    // ---- Load 8 bf16 values ----
    bf16x8 x = bf16x8::load(x_ptr + 8 * (row * nvecs_per_row + tid));

    // ---- Step 1: per-row abs-max via shared-memory tree reduce ----
    nv_bfloat16 local_abs_max = vecReduceAbsMax(x);
    smem[tid] = local_abs_max;
    if (tid == nvecs_per_row - 1) smem[nvecs_per_row] = __float2bfloat16(0.0f);
    __syncthreads();

    for (int stride = nvecs_per_row / 2; stride >= 2; stride /= 2) {
        if (tid < stride) {
            nv_bfloat162 a = make_bfloat162(smem[tid],          smem[tid + 1]);
            nv_bfloat162 b = make_bfloat162(smem[tid + stride], smem[tid + stride + 1]);
            nv_bfloat162 m;
            __asm__ volatile(
                "max.xorsign.abs.bf16x2 %0, %1, %2;"
                : "=r"(reinterpret_cast<int&>(m))
                : "r"(reinterpret_cast<int&>(a)), "r"(reinterpret_cast<int&>(b)));
            smem[tid]     = m.x;
            smem[tid + 1] = m.y;
        }
        __syncthreads();
    }

    if (tid == 0) {
        // For non-power-of-2 nvecs_per_row (e.g. 96 for k=768), the stride loop exits
        // with `last_stride` unmerged elements in smem[0..last_stride-1].
        // For power-of-2 this is always 2 (smem[0,1]), matching the original behaviour.
        int last_stride = nvecs_per_row / 2;
        while (last_stride / 2 >= 2) last_stride /= 2;
        nv_bfloat16 amax_bf = __habs(smem[0]);
        for (int j = 1; j < last_stride; j++)
            amax_bf = __hmax(amax_bf, __habs(smem[j]));
        float        row_amax = static_cast<float>(amax_bf);

        constexpr float inv_scales_max = (NumCandidates > 1) ? (1.f / 256.f) : (1.f / 448.f);
        constexpr float one_over_six   = 1.f / 6.f;
        float s = (row_amax == 0.f)
                ? 1.f
                : row_amax * inv_scales_max * (scale_override * one_over_six);
        global_scale_ptr[row]         = s;
        *reinterpret_cast<float*>(smem) = s;
    }
    __syncthreads();
    float scale = *reinterpret_cast<float*>(smem);

    // ---- Step 2: per-group (16-element) abs-max ----
    nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
    float full_abs_max = static_cast<float>(
        __hmax(__habs(local_abs_max), __habs(other_abs_max)));

    // ---- Step 3: quantize (RTN or Round-4-or-6) ----
    QuantResultGP res;
    if constexpr (NumCandidates == 1) {
        res = quantize_gp(full_abs_max, (1.f / 6.f) * scale_override, scale, x);
    } else {
        float candidates[2] = {6.f, 4.f};
        QuantResultGP results[2];
        float scores[2];
        float best = __builtin_inff();
        for (int i = 0; i < 2; ++i) {
            results[i] = quantize_gp(full_abs_max, (1.f / candidates[i]) * scale_override, scale, x);
            // Compute L2 error (same warp-level reduction as single-group kernel)
            const float descale = static_cast<float>(results[i].fp8s) * scale;
            float2 sum = {0.f, 0.f};
            const float2 dsv = {-descale, -descale};
            for (int j = 0; j < 4; ++j) {
                float2 dq = __nv_cvt_fp4x2_to_float2(results[i].bits[j]);
                float2 xv = {static_cast<float>(x[2*j]), static_cast<float>(x[2*j+1])};
                float2 d  = __ffma2_rn(dq, dsv, xv);
                sum = __ffma2_rn(d, d, sum);
            }
            float local_err = sum.x + sum.y;
            local_err += __shfl_xor_sync(0xffffffff, local_err, 1);
            scores[i] = local_err;
            if (scores[i] < best) { best = scores[i]; res = results[i]; }
        }
    }

    // ---- Step 4: store packed FP4 (row-major, global row index) ----
    const int global_vec_idx = row * nvecs_per_row + tid;
    res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * global_vec_idx);

    // ---- Step 5: store per-group FP8 scale with per-group swizzle ----
    // Even threads only (each thread covers 2 adjacent cols → one 16-element group)
    if (tid % 2 == 0) {
        const int   col       = tid / 2;
        const int   numKTiles = cols_div16 / 4;  // cols / 64
        const int64_t tgt = cvt_quant_to_fp4_get_sf_out_offset(local_row, col, numKTiles);
        scale_ptr[sf_base + tgt] = res.fp8s;
    }
}

// ---- Launch helper ----

static void launch_fp4_group_pertoken(
    int num_candidates,
    __nv_fp4x4_e2m1*   y_ptr,
    __nv_fp8_e4m3*     scale_ptr,
    float*             global_scale_ptr,
    const nv_bfloat16* x_ptr,
    const int32_t*     row_offsets,
    const int64_t*     scale_offsets,
    float  scale_override,
    long   total_rows,
    long   cols,
    int    num_groups,
    cudaStream_t stream)
{
    int    threads = static_cast<int>(cols / 8);
    size_t smem    = (static_cast<size_t>(threads) + 1) * sizeof(nv_bfloat16);
    int    cdiv16  = static_cast<int>(cols / 16);

    if (num_candidates == 2)
        fp4_group_pertoken_kernel<2><<<total_rows, threads, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups);
    else
        fp4_group_pertoken_kernel<1><<<total_rows, threads, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups);

    CUDA_CHECK(cudaGetLastError());
}

// ---- PyTorch bindings ----

void rtn_fp4_group_pertoken_binding(
    at::Tensor&       out,           // [total_m, cols/2]  uint8
    at::Tensor&       scales,        // flat FP8 scale buffer, pre-allocated with zeros
    at::Tensor&       global_scale,  // [total_m]  float32
    const at::Tensor& inp,           // [total_m, cols]  bfloat16
    const at::Tensor& row_offsets,   // [num_groups+1]  int32  (on GPU)
    const at::Tensor& scale_offsets, // [num_groups+1]  int64  (on GPU)
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()   == at::kBFloat16,  "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                   "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),             "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                  "inp must be 2D [total_m, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,          "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 512,          "cols must be <= 4096");
    TORCH_CHECK(global_scale.numel() == inp.size(0), "global_scale size mismatch");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt,  "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong, "scale_offsets must be int64");

    const long total_m = inp.size(0);
    const long cols    = inp.size(1);
    const int  ngroups = static_cast<int>(row_offsets.numel()) - 1;

    launch_fp4_group_pertoken(
        1,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        scale_override, total_m, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}

void four_six_fp4_group_pertoken_binding(
    at::Tensor&       out,
    at::Tensor&       scales,
    at::Tensor&       global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()   == at::kBFloat16,  "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                   "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),             "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                  "inp must be 2D [total_m, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,          "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 512,          "cols must be <= 4096");
    TORCH_CHECK(global_scale.numel() == inp.size(0), "global_scale size mismatch");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt,  "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong, "scale_offsets must be int64");

    const long total_m = inp.size(0);
    const long cols    = inp.size(1);
    const int  ngroups = static_cast<int>(row_offsets.numel()) - 1;

    launch_fp4_group_pertoken(
        2,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        scale_override, total_m, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}
