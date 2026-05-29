// Per-token NV FP4 grouped quantization with per-group tcgen05 swizzled SF layout.
//
// Extends fp4_pertoken_kernel to handle multiple row-groups in one kernel launch.
// Each group's scale factor is independently swizzled (local row index within the group),
// so the output is directly usable by the CuTeDSL grouped GEMM without any CPU layout
// conversion (eliminating the D2H → CPU CuTeDSL kernel → H2D path).
//
// Grid/block assignment (mirrors quantize_pertoken_fp4_fused.cu):
//   nvecs_per_row  = cols / 8
//   warps_per_row  = ceil(nvecs_per_row / 32)
//   rows_per_block = max(1, 1024 / (warps_per_row * 32))
//   blockDim       = rows_per_block * warps_per_row * 32
//   natural_grid   = ceil(total_rows / rows_per_block)
//   gridDim        = min(natural_grid, MAX_BLOCKS)        ← capped
//
// Each block processes rows_per_block rows; a grid-stride outer loop handles the
// case where natural_grid > MAX_BLOCKS.
//
// Shared memory layout (pertoken kernel):
//   bf16[rows_per_block * (nvecs_per_row + 1)]
//   row r uses smem[r * (nvecs_per_row + 1) .. r * (nvecs_per_row + 1) + nvecs_per_row]
//   The +1 sentinel slot is required by the tree-reduce when nvecs_per_row is odd.
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

// global_encode_scale = 2688.0f / row_abs_max  (matches golden's formula exactly)
// scale               = scale_override / global_encode_scale
__device__ __forceinline__ QuantResultGP
quantize_gp(float abs_max, float val_max, float global_encode_scale, float scale, bf16x8& x)
{
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(fminf(abs_max * (1.0f / val_max) * global_encode_scale, 448.0f));
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

// Same quantization logic as quantize() in quantize_round_four_six.cu.
// Uses reciprocal_approximate_ftz for bit-exact match with rtn_fp4 / four_six_fp4.
// inv_val_max = scale_override / val_max  (e.g. 1/6 for RTN, 1/4 for 4-or-6)
__device__ __forceinline__ QuantResultGP
quantize_tg(float abs_max, float inv_val_max, float scale, bf16x8& x)
{
    float s_group    = abs_max * inv_val_max;
    float inv_scale  = reciprocal_approximate_ftz(scale);
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group * inv_scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0.f) s_round_fp8 = 1.f;

    float factor   = reciprocal_approximate_ftz(s_round_fp8 * scale);
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
//
// Shared memory layout (two independent regions to guarantee float alignment):
//   Region A (bf16): rows_per_block * (nvecs_per_row + 1) * sizeof(bf16)
//     row r uses bf16_smem[r * (nvecs_per_row + 1) .. + nvecs_per_row]
//   Region B (float, 4-byte aligned): rows_per_block * 2 * sizeof(float)
//     row r uses float_smem[r * 2]   = scale
//              float_smem[r * 2 + 1] = global_encode_scale
//   Region A size is rounded up to 4-byte boundary before Region B starts.
template<int NumCandidates>
__global__ void fp4_group_pertoken_kernel(
    __nv_fp4x4_e2m1*    __restrict__ y_ptr,
    __nv_fp8_e4m3*      __restrict__ scale_ptr,
    float*              __restrict__ global_scale_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ row_offsets,    // [num_groups+1]
    const int64_t*      __restrict__ scale_offsets,  // [num_groups+1]
    float  scale_override,
    int    cols_div16,        // cols / 16
    int    num_groups,
    int    nvecs_per_row,     // cols / 8
    int    warps_per_row,
    int    rows_per_block,
    int    total_rows,
    int    grid_stride,       // gridDim.x * rows_per_block
    int    bf16_region_floats)// byte offset (in floats) where Region B starts
{
    extern __shared__ float smem_raw[];
    // Region A: bf16 abs-max reduce buffers
    nv_bfloat16* bf16_smem = reinterpret_cast<nv_bfloat16*>(smem_raw);
    // Region B: float broadcast (scale, ges) — starts at 4-byte aligned offset
    float*        f_smem   = smem_raw + bf16_region_floats;

    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid / 32;
    const int lane_id     = flat_tid % 32;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int tid         = warp_in_row * 32 + lane_id;

    const int smem_base = local_row * (nvecs_per_row + 1);  // bf16 slot base for this row

    // Outer grid-stride loop.
    for (int base_row = blockIdx.x * rows_per_block;
         base_row < total_rows;
         base_row += grid_stride)
    {
        const int row        = base_row + local_row;
        const bool valid_row = row < total_rows;
        const bool valid_vec = valid_row && (tid < nvecs_per_row);

        // ---- Find which group owns this row ----
        int group_id = 0;
        if (valid_row) {
            while (group_id < num_groups - 1 && row >= row_offsets[group_id + 1])
                ++group_id;
        }
        const int     local_row_in_group = valid_row ? row - row_offsets[group_id] : 0;
        const int64_t sf_base            = valid_row ? scale_offsets[group_id] : 0;

        // ---- Load 8 bf16 values ----
        bf16x8 x{};
        nv_bfloat16 local_abs_max = __float2bfloat16(0.f);
        if (valid_vec) {
            x = bf16x8::load(x_ptr + 8 * (row * nvecs_per_row + tid));
            local_abs_max = vecReduceAbsMax(x);
        }

        // ---- Step 1: per-row abs-max via shared-memory tree reduce ----
        bf16_smem[smem_base + tid] = valid_vec ? local_abs_max : __float2bfloat16(0.f);
        if (tid == nvecs_per_row - 1 || (!valid_vec && tid == nvecs_per_row))
            bf16_smem[smem_base + nvecs_per_row] = __float2bfloat16(0.f);
        __syncthreads();

        for (int stride = nvecs_per_row / 2; stride >= 2; stride /= 2) {
            if (valid_row && tid < stride) {
                nv_bfloat162 a = make_bfloat162(bf16_smem[smem_base + tid],          bf16_smem[smem_base + tid + 1]);
                nv_bfloat162 b = make_bfloat162(bf16_smem[smem_base + tid + stride], bf16_smem[smem_base + tid + stride + 1]);
                nv_bfloat162 m;
                __asm__ volatile(
                    "max.xorsign.abs.bf16x2 %0, %1, %2;"
                    : "=r"(reinterpret_cast<int&>(m))
                    : "r"(reinterpret_cast<int&>(a)), "r"(reinterpret_cast<int&>(b)));
                bf16_smem[smem_base + tid]     = m.x;
                bf16_smem[smem_base + tid + 1] = m.y;
            }
            __syncthreads();
        }

        // tid==0 of each row: finish reduce, write scale + ges to Region B (float, aligned).
        if (valid_row && tid == 0) {
            int last_stride = nvecs_per_row / 2;
            while (last_stride / 2 >= 2) last_stride /= 2;
            nv_bfloat16 amax_bf = __habs(bf16_smem[smem_base]);
            for (int j = 1; j < last_stride; j++)
                amax_bf = __hmax(amax_bf, __habs(bf16_smem[smem_base + j]));
            float row_amax = static_cast<float>(amax_bf);

            constexpr float SCALES_MAX = (NumCandidates > 1) ? 256.0f : 448.0f;
            constexpr float FP4_MAX    = 6.0f;
            float ges = (row_amax == 0.f) ? 1.f : (SCALES_MAX * FP4_MAX) / row_amax;
            float s   = (row_amax == 0.f) ? scale_override : scale_override / ges;
            global_scale_ptr[row]        = s;
            f_smem[local_row * 2 + 0]   = s;
            f_smem[local_row * 2 + 1]   = ges;
        }
        __syncthreads();

        // ---- Step 2: per-group (16-element) abs-max ----
        // Must be called by ALL threads in the warp (before any continue that
        // would cause divergence). OOB threads have local_abs_max==0, which is
        // the neutral element for max, so valid threads get correct results.
        nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
        float full_abs_max = static_cast<float>(
            __hmax(__habs(local_abs_max), __habs(other_abs_max)));

        if (!valid_row) continue;

        const float scale               = f_smem[local_row * 2 + 0];
        const float global_encode_scale = f_smem[local_row * 2 + 1];

        if (!valid_vec) continue;

        // ---- Step 3: quantize (RTN or Round-4-or-6) ----
        QuantResultGP res;
        if constexpr (NumCandidates == 1) {
            res = quantize_gp(full_abs_max, 6.0f, global_encode_scale, scale, x);
        } else {
            float candidates[2] = {6.f, 4.f};
            QuantResultGP results[2];
            float scores[2];
            float best = __builtin_inff();
            for (int i = 0; i < 2; ++i) {
                results[i] = quantize_gp(full_abs_max, candidates[i], global_encode_scale, scale, x);
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

        // ---- Step 4: store packed FP4 ----
        const int global_vec_idx = row * nvecs_per_row + tid;
        res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * global_vec_idx);

        // ---- Step 5: store per-group FP8 scale with per-group swizzle ----
        if (tid % 2 == 0) {
            const int     col      = tid / 2;
            const int     numKTiles = cols_div16 / 4;
            const int64_t tgt = cvt_quant_to_fp4_get_sf_out_offset(local_row_in_group, col, numKTiles);
            scale_ptr[sf_base + tgt] = res.fp8s;
        }
    } // end grid-stride loop
}

static constexpr int MAX_BLOCKS_GP = 1024;

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
    const int nvecs_per_row  = static_cast<int>(cols / 8);
    const int warps_per_row  = (nvecs_per_row + 31) / 32;
    const int rows_per_block = max(1, 1024 / (warps_per_row * 32));
    const int block_dim      = rows_per_block * warps_per_row * 32;
    const int natural_grid   = (static_cast<int>(total_rows) + rows_per_block - 1) / rows_per_block;
    const int grid_dim       = min(natural_grid, MAX_BLOCKS_GP);
    const int grid_stride    = grid_dim * rows_per_block;
    const int cdiv16         = static_cast<int>(cols / 16);
    // Region A: rows_per_block * (nvecs_per_row + 1) bf16, rounded up to 4-byte boundary.
    // Region B: rows_per_block * 2 float (4-byte aligned, starts right after Region A).
    const size_t bf16_bytes       = static_cast<size_t>(rows_per_block) * (nvecs_per_row + 1) * sizeof(nv_bfloat16);
    const size_t bf16_bytes_align = (bf16_bytes + 3) & ~size_t(3);  // round up to 4 bytes
    const int    bf16_region_floats = static_cast<int>(bf16_bytes_align / sizeof(float));
    const size_t smem             = bf16_bytes_align + static_cast<size_t>(rows_per_block) * 2 * sizeof(float);

    if (num_candidates == 2)
        fp4_group_pertoken_kernel<2><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block, total_rows, grid_stride,
            bf16_region_floats);
    else
        fp4_group_pertoken_kernel<1><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block, total_rows, grid_stride,
            bf16_region_floats);

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

// ── Grouped tensor-global FP4 quantization ────────────────────────────────────
//
// Like fp4_group_pertoken_kernel but the amax is per-group (tensor-global),
// not per-row.  Each block reads amax_ptr[group_id] instead of doing a
// shared-memory row reduce.  This makes it suitable for weight quantization
// where all rows in a group (one MoE expert) share a single amax.
//
// Grid/block assignment follows the same pattern as fp4_group_pertoken_kernel.
// Shared memory: float[rows_per_block * 2] — [scale, global_encode_scale] per row.

template<int NumCandidates>
__global__ void fp4_group_tensor_global_kernel(
    __nv_fp4x4_e2m1*    __restrict__ y_ptr,
    __nv_fp8_e4m3*      __restrict__ scale_ptr,
    float*              __restrict__ global_scale_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ row_offsets,    // [num_groups+1]
    const int64_t*      __restrict__ scale_offsets,  // [num_groups+1]
    const float*        __restrict__ amax_ptr,       // [num_groups]
    float  scale_override,
    int    cols_div16,
    int    num_groups,
    int    nvecs_per_row,
    int    warps_per_row,
    int    rows_per_block,
    int    total_rows,
    int    grid_stride)
{
    // smem_f[local_row * 2 + 0] = scale, smem_f[local_row * 2 + 1] = global_encode_scale
    extern __shared__ float smem_f[];

    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid / 32;
    const int lane_id     = flat_tid % 32;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int tid         = warp_in_row * 32 + lane_id;

    for (int base_row = blockIdx.x * rows_per_block;
         base_row < total_rows;
         base_row += grid_stride)
    {
        const int  row       = base_row + local_row;
        const bool valid_row = row < total_rows;
        const bool valid_vec = valid_row && (tid < nvecs_per_row);

        // Find group owning this row.
        int group_id = 0;
        if (valid_row) {
            while (group_id < num_groups - 1 && row >= row_offsets[group_id + 1])
                ++group_id;
        }
        const int     local_row_in_group = valid_row ? row - row_offsets[group_id] : 0;
        const int64_t sf_base            = valid_row ? scale_offsets[group_id] : 0;

        // tid==0 of each row broadcasts per-group scale into smem.
        if (valid_row && tid == 0) {
            float group_amax = amax_ptr[group_id];
            constexpr float inv_scales_max = (NumCandidates > 1) ? (1.f / 256.f) : (1.f / 448.f);
            constexpr float inv_six        = 1.f / 6.f;
            float inv_val_max = scale_override * inv_six;
            float s = (group_amax == 0.f) ? 1.f : group_amax * inv_scales_max * inv_val_max;
            // Write global_scale once per group (on the first row of the group that this
            // CTA processes; subsequent grid-stride visits to the same group are no-ops
            // since the value is identical).
            if (local_row_in_group == 0)
                global_scale_ptr[group_id] = s;
            smem_f[local_row * 2 + 0] = s;
        }
        __syncthreads();

        if (!valid_row) continue;

        const float scale = smem_f[local_row * 2 + 0];

        // Load 8 bf16 values (OOB threads load zero).
        bf16x8 x{};
        nv_bfloat16 local_abs_max = __float2bfloat16(0.f);
        if (valid_vec) {
            x = bf16x8::load(x_ptr + 8 * (row * nvecs_per_row + tid));
            local_abs_max = vecReduceAbsMax(x);
        }

        // Per-group-of-16 abs-max: must be called by all warp threads uniformly.
        // OOB threads contribute 0 (neutral for max), so valid threads are unaffected.
        nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
        float full_abs_max = static_cast<float>(__hmax(local_abs_max, other_abs_max));

        if (!valid_vec) continue;

        // Quantize using reciprocal_approximate_ftz path (bit-exact with rtn_fp4).
        QuantResultGP res;
        if constexpr (NumCandidates == 1) {
            constexpr float inv_six = 1.f / 6.f;
            res = quantize_tg(full_abs_max, scale_override * inv_six, scale, x);
        } else {
            float candidates[2]    = {6.f, 4.f};
            QuantResultGP results[2];
            float scores[2];
            float best = __builtin_inff();
            for (int i = 0; i < 2; ++i) {
                float inv_cand = 1.f / candidates[i];
                results[i] = quantize_tg(full_abs_max, scale_override * inv_cand, scale, x);
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

        // Store packed FP4.
        const int global_vec_idx = row * nvecs_per_row + tid;
        res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * global_vec_idx);

        // Store FP8 scale with per-group swizzle.
        if (tid % 2 == 0) {
            const int     col      = tid / 2;
            const int     numKTiles = cols_div16 / 4;
            const int64_t tgt = cvt_quant_to_fp4_get_sf_out_offset(local_row_in_group, col, numKTiles);
            scale_ptr[sf_base + tgt] = res.fp8s;
        }
    } // end grid-stride loop
}

static void launch_fp4_group_tensor_global(
    int num_candidates,
    __nv_fp4x4_e2m1*   y_ptr,
    __nv_fp8_e4m3*     scale_ptr,
    float*             global_scale_ptr,
    const nv_bfloat16* x_ptr,
    const int32_t*     row_offsets,
    const int64_t*     scale_offsets,
    const float*       amax_ptr,
    float  scale_override,
    long   total_rows,
    long   cols,
    int    num_groups,
    cudaStream_t stream)
{
    const int nvecs_per_row  = static_cast<int>(cols / 8);
    const int warps_per_row  = (nvecs_per_row + 31) / 32;
    const int rows_per_block = max(1, 1024 / (warps_per_row * 32));
    const int block_dim      = rows_per_block * warps_per_row * 32;
    const int natural_grid   = (static_cast<int>(total_rows) + rows_per_block - 1) / rows_per_block;
    const int grid_dim       = min(natural_grid, MAX_BLOCKS_GP);
    const int grid_stride    = grid_dim * rows_per_block;
    const int cdiv16         = static_cast<int>(cols / 16);
    // smem: rows_per_block * 2 floats (scale + ges per row).
    const size_t smem = static_cast<size_t>(rows_per_block) * 2 * sizeof(float);

    if (num_candidates == 2)
        fp4_group_tensor_global_kernel<2><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, amax_ptr, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block, total_rows, grid_stride);
    else
        fp4_group_tensor_global_kernel<1><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, amax_ptr, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block, total_rows, grid_stride);

    CUDA_CHECK(cudaGetLastError());
}

void rtn_fp4_group_tensor_global_binding(
    at::Tensor&       out,           // [total_n, cols/2]  uint8
    at::Tensor&       scales,        // flat FP8 scale buffer, pre-zeroed
    at::Tensor&       global_scale,  // [num_groups]  float32
    const at::Tensor& inp,           // [total_n, cols]  bfloat16  (experts stacked)
    const at::Tensor& row_offsets,   // [num_groups+1]  int32  (on GPU)
    const at::Tensor& scale_offsets, // [num_groups+1]  int64  (on GPU)
    const at::Tensor& amax,          // [num_groups]  float32  (per-expert abs-max)
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()      == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                     "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),               "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                    "inp must be 2D [total_n, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,            "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 512,            "cols must be <= 4096");
    TORCH_CHECK(amax.dtype()     == at::kFloat,    "amax must be float32");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt, "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong,"scale_offsets must be int64");
    const int ngroups = static_cast<int>(row_offsets.numel()) - 1;
    TORCH_CHECK(global_scale.numel() == ngroups,   "global_scale must have num_groups elements");
    TORCH_CHECK(amax.numel()         == ngroups,   "amax must have num_groups elements");

    const long total_n = inp.size(0);
    const long cols    = inp.size(1);

    launch_fp4_group_tensor_global(
        1,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        reinterpret_cast<const float*>(amax.data_ptr()),
        scale_override, total_n, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}

void four_six_fp4_group_tensor_global_binding(
    at::Tensor&       out,
    at::Tensor&       scales,
    at::Tensor&       global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    const at::Tensor& amax,
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()      == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                     "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),               "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                    "inp must be 2D [total_n, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,            "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 512,            "cols must be <= 4096");
    TORCH_CHECK(amax.dtype()     == at::kFloat,    "amax must be float32");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt, "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong,"scale_offsets must be int64");
    const int ngroups = static_cast<int>(row_offsets.numel()) - 1;
    TORCH_CHECK(global_scale.numel() == ngroups,   "global_scale must have num_groups elements");
    TORCH_CHECK(amax.numel()         == ngroups,   "amax must have num_groups elements");

    const long total_n = inp.size(0);
    const long cols    = inp.size(1);

    launch_fp4_group_tensor_global(
        2,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        reinterpret_cast<const float*>(amax.data_ptr()),
        scale_override, total_n, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}
