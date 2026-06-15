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
#include <vector>

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
// Legacy per-row scheduler — preserved for benchmark comparison vs the new
// per-CTA-chunk + binary-search scheduler. Kept reachable through
// rtn_fp4_group_pertoken_legacy_per_row_binding / four_six_* (bench only).
// The public bindings rtn_fp4_group_pertoken / four_six_fp4_group_pertoken
// now route to the chunked launcher (defined further below).
//
// Shared memory layout (two independent regions to guarantee float alignment):
//   Region A (bf16): rows_per_block * (nvecs_per_row + 1) * sizeof(bf16)
//     row r uses bf16_smem[r * (nvecs_per_row + 1) .. + nvecs_per_row]
//   Region B (float, 4-byte aligned): rows_per_block * 2 * sizeof(float)
//     row r uses float_smem[r * 2]   = scale
//              float_smem[r * 2 + 1] = global_encode_scale
//   Region A size is rounded up to 4-byte boundary before Region B starts.
template<int NumCandidates>
__global__ void fp4_group_pertoken_legacy_per_row_kernel(
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

// ---- Launch helper (legacy per-row scheduler) ----

static void launch_fp4_group_pertoken_legacy_per_row(
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
        fp4_group_pertoken_legacy_per_row_kernel<2><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block, total_rows, grid_stride,
            bf16_region_floats);
    else
        fp4_group_pertoken_legacy_per_row_kernel<1><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block, total_rows, grid_stride,
            bf16_region_floats);

    CUDA_CHECK(cudaGetLastError());
}

// ── Per-CTA-chunk pertoken kernel (no fused pad) ──────────────────────────────
//
// Mirrors fp4_group_pertoken_pad_fuse_kernel's scheduling, but expects the
// input x to already be padded row-for-row to the output buffer — i.e. every
// per-group M is a multiple of CHUNK_M=128 and x.size(0) == out.size(0) ==
// total_M (no separate unpadded buffer).
//
//   - CHUNK_M=128 rows are owned by one CTA; thread-0 does an O(log G) binary
//     search on row_offsets to find the owning group and broadcasts
//     (row_base_in_group, sf_base) via __shared__.
//   - Eliminates the O(G) per-thread linear scan that the legacy per-row
//     kernel pays once per row.
//   - Preserves the exact bf16x8 load → tree-reduce → 16-elem-warp-pair amax
//     → quantize_gp(rtn|four-six) → swizzled SF store pipeline of the legacy
//     kernel, so output is bit-identical when total_M is 128-aligned.
//
// Precondition: total_rows % CHUNK_M == 0  (each m_g must itself be 128-aligned
// upstream — the wrapper / pad_fuse path guarantees this). TORCH_CHECK in the
// binding enforces it.
template<int NumCandidates>
__global__ void fp4_group_pertoken_kernel(
    __nv_fp4x4_e2m1*    __restrict__ y_ptr,
    __nv_fp8_e4m3*      __restrict__ scale_ptr,
    float*              __restrict__ global_scale_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ row_offsets,    // [num_groups+1], 128-aligned
    const int64_t*      __restrict__ scale_offsets,  // [num_groups+1]
    float  scale_override,
    int    cols_div16,
    int    num_groups,
    int    nvecs_per_row,
    int    warps_per_row,
    int    rows_per_block,      // power-of-2 divisor of CHUNK_M=128
    int    total_chunks,        // total_rows / 128
    int    grid_stride_chunks,  // = gridDim.x
    int    bf16_region_floats)
{
    constexpr int CHUNK_M = 128;

    extern __shared__ float smem_raw[];
    nv_bfloat16* bf16_smem = reinterpret_cast<nv_bfloat16*>(smem_raw);
    float*        f_smem   = smem_raw + bf16_region_floats;

    __shared__ int     s_row_base_in_group;
    __shared__ int64_t s_sf_base;

    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid / 32;
    const int lane_id     = flat_tid % 32;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int tid         = warp_in_row * 32 + lane_id;

    const int smem_base       = local_row * (nvecs_per_row + 1);
    const int iters_per_chunk = CHUNK_M / rows_per_block;

    for (int chunk_idx = blockIdx.x;
         chunk_idx < total_chunks;
         chunk_idx += grid_stride_chunks)
    {
        const int chunk_row_start = chunk_idx * CHUNK_M;  // global row index

        // thread-0: O(log G) binary search → broadcast owning-group context.
        if (flat_tid == 0) {
            int lo = 0, hi = num_groups - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (row_offsets[mid] <= chunk_row_start) lo = mid;
                else                                     hi = mid - 1;
            }
            const int g = lo;
            s_row_base_in_group = chunk_row_start - row_offsets[g];
            s_sf_base           = scale_offsets[g];
        }
        __syncthreads();

        const int     row_base = s_row_base_in_group;
        const int64_t sf_base  = s_sf_base;

        // Inner loop: iters_per_chunk × rows_per_block = CHUNK_M=128 rows.
        for (int iter = 0; iter < iters_per_chunk; ++iter) {
            const int row_in_chunk      = iter * rows_per_block + local_row;   // 0..127
            const int row_in_group      = row_base + row_in_chunk;
            const int row_global        = chunk_row_start + row_in_chunk;
            const bool valid_vec        = (tid < nvecs_per_row);

            // ---- Load 8 bf16 values ----
            bf16x8 x{};
            nv_bfloat16 local_abs_max = __float2bfloat16(0.f);
            if (valid_vec) {
                x = bf16x8::load(x_ptr + 8 * (row_global * nvecs_per_row + tid));
                local_abs_max = vecReduceAbsMax(x);
            }

            // ---- Step 1: per-row abs-max via shared-memory tree reduce ----
            bf16_smem[smem_base + tid] = valid_vec ? local_abs_max : __float2bfloat16(0.f);
            if (tid == nvecs_per_row - 1 || (!valid_vec && tid == nvecs_per_row))
                bf16_smem[smem_base + nvecs_per_row] = __float2bfloat16(0.f);
            __syncthreads();

            for (int stride = nvecs_per_row / 2; stride >= 2; stride /= 2) {
                if (tid < stride) {
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

            // tid==0 of each row: finish reduce, broadcast (s, ges).
            if (tid == 0) {
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
                global_scale_ptr[row_global] = s;
                f_smem[local_row * 2 + 0]   = s;
                f_smem[local_row * 2 + 1]   = ges;
            }
            __syncthreads();

            // ---- Step 2: per-group (16-element) abs-max (warp-uniform) ----
            nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
            float full_abs_max = static_cast<float>(
                __hmax(__habs(local_abs_max), __habs(other_abs_max)));

            const float scale               = f_smem[local_row * 2 + 0];
            const float global_encode_scale = f_smem[local_row * 2 + 1];

            if (valid_vec) {
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
                const int global_vec_idx = row_global * nvecs_per_row + tid;
                res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * global_vec_idx);

                // ---- Step 5: store per-group FP8 SF with swizzle ----
                if (tid % 2 == 0) {
                    const int     col       = tid / 2;
                    const int     numKTiles = cols_div16 / 4;
                    const int64_t tgt = cvt_quant_to_fp4_get_sf_out_offset(row_in_group, col, numKTiles);
                    scale_ptr[sf_base + tgt] = res.fp8s;
                }
            }
        } // end iter

        __syncthreads();  // ensure all threads finish before next chunk overwrites s_*
    } // end chunk loop
}

// ---- Launch helper (per-CTA-chunk + binary search) ----

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
    constexpr int CHUNK_M = 128;
    TORCH_CHECK(total_rows % CHUNK_M == 0,
                "fp4_group_pertoken (chunked): total_rows must be a multiple of 128 "
                "(each per-group M must be 128-aligned; the upstream Fp8Padding / wrapper "
                "guarantees this — got total_rows=", total_rows, ")");

    const int nvecs_per_row  = static_cast<int>(cols / 8);
    const int warps_per_row  = (nvecs_per_row + 31) / 32;
    const int raw_rpb        = max(1, 1024 / (warps_per_row * 32));
    // rows_per_block must be a power-of-2 divisor of CHUNK_M (=128).
    int rows_per_block = 1;
    while (rows_per_block * 2 <= raw_rpb && rows_per_block * 2 <= CHUNK_M)
        rows_per_block *= 2;

    const int block_dim     = rows_per_block * warps_per_row * 32;
    const int total_chunks  = static_cast<int>(total_rows / CHUNK_M);
    const int grid_dim      = min(total_chunks, MAX_BLOCKS_GP);
    const int cdiv16        = static_cast<int>(cols / 16);

    // Same smem layout as the legacy per-row kernel: Region A bf16 + Region B float.
    const size_t bf16_bytes         = static_cast<size_t>(rows_per_block) * (nvecs_per_row + 1) * sizeof(nv_bfloat16);
    const size_t bf16_bytes_align   = (bf16_bytes + 3) & ~size_t(3);
    const int    bf16_region_floats = static_cast<int>(bf16_bytes_align / sizeof(float));
    const size_t smem               = bf16_bytes_align + static_cast<size_t>(rows_per_block) * 2 * sizeof(float);

    if (num_candidates == 2)
        fp4_group_pertoken_kernel<2><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block,
            total_chunks, grid_dim, bf16_region_floats);
    else
        fp4_group_pertoken_kernel<1><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            row_offsets, scale_offsets, scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block,
            total_chunks, grid_dim, bf16_region_floats);

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

// ---- PyTorch bindings (legacy per-row scheduler, bench only) ----

void rtn_fp4_group_pertoken_legacy_per_row_binding(
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

    launch_fp4_group_pertoken_legacy_per_row(
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

void four_six_fp4_group_pertoken_legacy_per_row_binding(
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

    launch_fp4_group_pertoken_legacy_per_row(
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

// ── Per-CTA-chunk pertoken kernel with fused pad ─────────────────────────────
//
// Same data path as fp4_group_pertoken_kernel (Step 1..5), but the scheduling
// is per-CTA-chunk instead of per-row:
//
//   - Output / SF / global_scale buffers are sized for the *padded* total_M
//     (each group's M rounded up to CHUNK_M=128 — produced upstream by
//      Megatron Fp8Padding).
//   - The input bf16 x_ptr is *unpadded* (the kernel reads only the real rows).
//   - CHUNK_M=128 padded rows are owned by one CTA; thread-0 does a single
//     O(log G) binary search on padded_row_offsets to find the owning group
//     and broadcasts (padded_base_in_group, unpadded_base_global, m_unpadded,
//     sf_base) via __shared__ to every other thread (eliminates the O(G)
//     per-thread scan and the cost of materializing a padded bf16 view).
//   - Pad rows (padded_row_in_group >= m_unpadded) skip the bf16 load
//     (their abs_max stays 0); quantize_gp then naturally produces fp8s=0
//     and bits=0 → output FP4 / SF for pad rows is exact zero. f(0)=0 holds
//     for all common activations, so downstream MLP propagates zeros through.
//
// Precondition: padded_total_M (= out.size(0)) must be divisible by CHUNK_M=128,
//               and each group's *padded* M must be 128-aligned (Megatron
//               Fp8Padding satisfies both; the chunked kernel ERRORs out via
//               TORCH_CHECK if not).
template<int NumCandidates>
__global__ void fp4_group_pertoken_pad_fuse_kernel(
    __nv_fp4x4_e2m1*    __restrict__ y_ptr,
    __nv_fp8_e4m3*      __restrict__ scale_ptr,
    float*              __restrict__ global_scale_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,                  // [unpadded_total_M, cols]
    const int32_t*      __restrict__ padded_row_offsets,     // [num_groups+1], 128-aligned
    const int32_t*      __restrict__ unpadded_row_offsets,   // [num_groups+1]
    const int64_t*      __restrict__ scale_offsets,          // [num_groups+1]
    float  scale_override,
    int    cols_div16,
    int    num_groups,
    int    nvecs_per_row,
    int    warps_per_row,
    int    rows_per_block,      // power-of-2 divisor of CHUNK_M=128
    int    total_chunks,        // padded_total_M / 128
    int    grid_stride_chunks,  // = gridDim.x
    int    bf16_region_floats)
{
    constexpr int CHUNK_M = 128;

    extern __shared__ float smem_raw[];
    nv_bfloat16* bf16_smem = reinterpret_cast<nv_bfloat16*>(smem_raw);
    float*        f_smem   = smem_raw + bf16_region_floats;

    __shared__ int     s_padded_base_in_group;
    __shared__ int     s_m_unpadded;
    __shared__ int     s_unpadded_base_global;
    __shared__ int64_t s_sf_base;

    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid / 32;
    const int lane_id     = flat_tid % 32;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int tid         = warp_in_row * 32 + lane_id;

    const int smem_base       = local_row * (nvecs_per_row + 1);
    const int iters_per_chunk = CHUNK_M / rows_per_block;

    for (int chunk_idx = blockIdx.x;
         chunk_idx < total_chunks;
         chunk_idx += grid_stride_chunks)
    {
        const int chunk_row_start = chunk_idx * CHUNK_M;  // padded row index

        // thread-0: O(log G) binary search → broadcast owning-group context.
        if (flat_tid == 0) {
            int lo = 0, hi = num_groups - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (padded_row_offsets[mid] <= chunk_row_start) lo = mid;
                else                                            hi = mid - 1;
            }
            const int g = lo;
            s_padded_base_in_group = chunk_row_start - padded_row_offsets[g];
            s_unpadded_base_global = unpadded_row_offsets[g];
            s_m_unpadded           = unpadded_row_offsets[g + 1] - unpadded_row_offsets[g];
            s_sf_base              = scale_offsets[g];
        }
        __syncthreads();

        const int     padded_base = s_padded_base_in_group;
        const int     m_unpadded  = s_m_unpadded;
        const int     x_base      = s_unpadded_base_global;
        const int64_t sf_base     = s_sf_base;

        // Inner loop: iters_per_chunk × rows_per_block = CHUNK_M=128 rows.
        for (int iter = 0; iter < iters_per_chunk; ++iter) {
            const int row_in_chunk      = iter * rows_per_block + local_row;   // 0..127
            const int padded_row_in_grp = padded_base + row_in_chunk;
            const int padded_row_global = chunk_row_start + row_in_chunk;
            const bool is_real_row      = padded_row_in_grp < m_unpadded;
            const bool valid_vec        = (tid < nvecs_per_row);
            const bool can_load         = is_real_row && valid_vec;

            // ---- Load 8 bf16 values (only from real rows; pad rows keep x=0) ----
            bf16x8 x{};
            nv_bfloat16 local_abs_max = __float2bfloat16(0.f);
            if (can_load) {
                const int unpadded_row = x_base + padded_row_in_grp;
                x = bf16x8::load(x_ptr + 8 * (unpadded_row * nvecs_per_row + tid));
                local_abs_max = vecReduceAbsMax(x);
            }

            // ---- Step 1: per-row abs-max via shared-memory tree reduce ----
            bf16_smem[smem_base + tid] = valid_vec ? local_abs_max : __float2bfloat16(0.f);
            if (tid == nvecs_per_row - 1 || (!valid_vec && tid == nvecs_per_row))
                bf16_smem[smem_base + nvecs_per_row] = __float2bfloat16(0.f);
            __syncthreads();

            for (int stride = nvecs_per_row / 2; stride >= 2; stride /= 2) {
                if (tid < stride) {
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

            // tid==0 of each row: finish reduce, broadcast (s, ges).
            if (tid == 0) {
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
                global_scale_ptr[padded_row_global] = s;
                f_smem[local_row * 2 + 0]   = s;
                f_smem[local_row * 2 + 1]   = ges;
            }
            __syncthreads();

            // ---- Step 2: per-group (16-element) abs-max (warp-uniform) ----
            nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
            float full_abs_max = static_cast<float>(
                __hmax(__habs(local_abs_max), __habs(other_abs_max)));

            const float scale               = f_smem[local_row * 2 + 0];
            const float global_encode_scale = f_smem[local_row * 2 + 1];

            // For pad rows: full_abs_max==0 → quantize_gp returns bits=0,fp8s=0;
            // the writes below put exact zeros in FP4 data + SF buffers.
            if (valid_vec) {
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

                // ---- Step 4: store packed FP4 (padded layout) ----
                const int global_vec_idx = padded_row_global * nvecs_per_row + tid;
                res.bits.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * global_vec_idx);

                // ---- Step 5: store per-group FP8 SF with swizzle ----
                if (tid % 2 == 0) {
                    const int     col       = tid / 2;
                    const int     numKTiles = cols_div16 / 4;
                    const int64_t tgt = cvt_quant_to_fp4_get_sf_out_offset(padded_row_in_grp, col, numKTiles);
                    scale_ptr[sf_base + tgt] = res.fp8s;
                }
            }
        } // end iter

        __syncthreads();  // ensure all threads finish before next chunk overwrites s_*
    } // end chunk loop
}

// ---- Launch helper (per-CTA-chunk + fused pad) ----

static void launch_fp4_group_pertoken_pad_fuse(
    int num_candidates,
    __nv_fp4x4_e2m1*   y_ptr,
    __nv_fp8_e4m3*     scale_ptr,
    float*             global_scale_ptr,
    const nv_bfloat16* x_ptr,
    const int32_t*     padded_row_offsets,
    const int32_t*     unpadded_row_offsets,
    const int64_t*     scale_offsets,
    float  scale_override,
    long   padded_total_m,
    long   cols,
    int    num_groups,
    cudaStream_t stream)
{
    constexpr int CHUNK_M = 128;
    TORCH_CHECK(padded_total_m % CHUNK_M == 0,
                "padded_total_m must be a multiple of 128 for the chunked kernel");

    const int nvecs_per_row  = static_cast<int>(cols / 8);
    const int warps_per_row  = (nvecs_per_row + 31) / 32;
    const int raw_rpb        = max(1, 1024 / (warps_per_row * 32));
    // rows_per_block must be a power-of-2 divisor of CHUNK_M (=128).
    int rows_per_block = 1;
    while (rows_per_block * 2 <= raw_rpb && rows_per_block * 2 <= CHUNK_M)
        rows_per_block *= 2;

    const int block_dim     = rows_per_block * warps_per_row * 32;
    const int total_chunks  = static_cast<int>(padded_total_m / CHUNK_M);
    const int grid_dim      = min(total_chunks, MAX_BLOCKS_GP);
    const int cdiv16        = static_cast<int>(cols / 16);

    // Same smem layout as the per-row kernel: Region A bf16 + Region B float.
    const size_t bf16_bytes         = static_cast<size_t>(rows_per_block) * (nvecs_per_row + 1) * sizeof(nv_bfloat16);
    const size_t bf16_bytes_align   = (bf16_bytes + 3) & ~size_t(3);
    const int    bf16_region_floats = static_cast<int>(bf16_bytes_align / sizeof(float));
    const size_t smem               = bf16_bytes_align + static_cast<size_t>(rows_per_block) * 2 * sizeof(float);

    if (num_candidates == 2)
        fp4_group_pertoken_pad_fuse_kernel<2><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            padded_row_offsets, unpadded_row_offsets, scale_offsets,
            scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block,
            total_chunks, grid_dim, bf16_region_floats);
    else
        fp4_group_pertoken_pad_fuse_kernel<1><<<grid_dim, block_dim, smem, stream>>>(
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,
            padded_row_offsets, unpadded_row_offsets, scale_offsets,
            scale_override, cdiv16, num_groups,
            nvecs_per_row, warps_per_row, rows_per_block,
            total_chunks, grid_dim, bf16_region_floats);

    CUDA_CHECK(cudaGetLastError());
}

// ---- PyTorch bindings (pad-fuse) ----

void rtn_fp4_group_pertoken_pad_fuse_binding(
    at::Tensor&       out,                    // [padded_total_m, cols/2]  uint8
    at::Tensor&       scales,                 // flat FP8 SF buffer (padded layout)
    at::Tensor&       global_scale,           // [padded_total_m]  float32
    const at::Tensor& inp,                    // [unpadded_total_m, cols]  bfloat16
    const at::Tensor& padded_row_offsets,     // [num_groups+1]  int32
    const at::Tensor& unpadded_row_offsets,   // [num_groups+1]  int32
    const at::Tensor& scale_offsets,          // [num_groups+1]  int64
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()   == at::kBFloat16,  "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                   "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),             "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                  "inp must be 2D [unpadded_total_m, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,          "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 512,          "cols must be <= 4096");
    TORCH_CHECK(out.dim() == 2,                  "out must be 2D [padded_total_m, cols/2]");
    TORCH_CHECK(out.size(0) % 128 == 0,          "padded_total_m (out.size(0)) must be 128-aligned");
    TORCH_CHECK(global_scale.numel() == out.size(0), "global_scale must size padded_total_m");
    TORCH_CHECK(padded_row_offsets.dtype()   == at::kInt,  "padded_row_offsets must be int32");
    TORCH_CHECK(unpadded_row_offsets.dtype() == at::kInt,  "unpadded_row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype()        == at::kLong, "scale_offsets must be int64");
    TORCH_CHECK(padded_row_offsets.numel()   == unpadded_row_offsets.numel(),
                "padded/unpadded_row_offsets size mismatch");

    const long padded_total_m = out.size(0);
    const long cols           = inp.size(1);
    const int  ngroups        = static_cast<int>(padded_row_offsets.numel()) - 1;

    launch_fp4_group_pertoken_pad_fuse(
        1,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(padded_row_offsets.data_ptr()),
        reinterpret_cast<const int32_t*>(unpadded_row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        scale_override, padded_total_m, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}

void four_six_fp4_group_pertoken_pad_fuse_binding(
    at::Tensor&       out,
    at::Tensor&       scales,
    at::Tensor&       global_scale,
    const at::Tensor& inp,
    const at::Tensor& padded_row_offsets,
    const at::Tensor& unpadded_row_offsets,
    const at::Tensor& scale_offsets,
    float             scale_override)
{
    TORCH_CHECK(inp.dtype()   == at::kBFloat16,  "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                   "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),             "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                  "inp must be 2D [unpadded_total_m, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,          "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 512,          "cols must be <= 4096");
    TORCH_CHECK(out.dim() == 2,                  "out must be 2D [padded_total_m, cols/2]");
    TORCH_CHECK(out.size(0) % 128 == 0,          "padded_total_m (out.size(0)) must be 128-aligned");
    TORCH_CHECK(global_scale.numel() == out.size(0), "global_scale must size padded_total_m");
    TORCH_CHECK(padded_row_offsets.dtype()   == at::kInt,  "padded_row_offsets must be int32");
    TORCH_CHECK(unpadded_row_offsets.dtype() == at::kInt,  "unpadded_row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype()        == at::kLong, "scale_offsets must be int64");
    TORCH_CHECK(padded_row_offsets.numel()   == unpadded_row_offsets.numel(),
                "padded/unpadded_row_offsets size mismatch");

    const long padded_total_m = out.size(0);
    const long cols           = inp.size(1);
    const int  ngroups        = static_cast<int>(padded_row_offsets.numel()) - 1;

    launch_fp4_group_pertoken_pad_fuse(
        2,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(padded_row_offsets.data_ptr()),
        reinterpret_cast<const int32_t*>(unpadded_row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        scale_override, padded_total_m, cols, ngroups,
        at::cuda::getCurrentCUDAStream());
}

// ── Grouped tensor-global FP4 quantization ────────────────────────────────────
//
// Grid/block assignment (group-aligned):
//   For each group g: ctas_g = ceil(m_g / rows_per_block)
//   group_cta_offsets[g] = prefix sum of ctas_g
//   gridDim = group_cta_offsets[num_groups]   (no cap — no grid-stride loop)
//   blockDim = rows_per_block * warps_per_row * 32
//
// Each CTA owns exactly one group: no inter-group row traversal.
// Binary search on group_cta_offsets replaces the O(G) linear scan per thread.
// amax is pre-computed by the caller (one Python op) and passed in directly.

// ---- shared helper: binary search to find which group owns blockIdx.x ----
__device__ __forceinline__ int tg_find_group(const int32_t* cta_offsets, int num_groups)
{
    // Find largest g such that cta_offsets[g] <= blockIdx.x.
    int lo = 0, hi = num_groups - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (cta_offsets[mid] <= static_cast<int>(blockIdx.x)) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// ---- Kernel: quantize using caller-supplied per-group amax -------------------
template<int NumCandidates, bool Swizzle>
__global__ void fp4_group_tensor_global_kernel(
    __nv_fp4x4_e2m1*    __restrict__ y_ptr,
    __nv_fp8_e4m3*      __restrict__ scale_ptr,
    float*              __restrict__ global_scale_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ group_cta_offsets,// [num_groups+1]
    const int32_t*      __restrict__ row_offsets,      // [num_groups+1]
    const int64_t*      __restrict__ scale_offsets,    // [num_groups+1]
    const float*        __restrict__ amax_ptr,         // [num_groups]
    float  scale_override,
    int    cols_div16,
    int    num_groups,
    int    nvecs_per_row,
    int    warps_per_row,
    int    rows_per_block)
{
    const int group_id    = tg_find_group(group_cta_offsets, num_groups);
    const int local_block = static_cast<int>(blockIdx.x) - group_cta_offsets[group_id];
    const int group_row0  = row_offsets[group_id];
    const int m_g         = row_offsets[group_id + 1] - group_row0;

    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid / 32;
    const int lane_id     = flat_tid % 32;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int tid         = warp_in_row * 32 + lane_id;

    const int  row_in_g  = local_block * rows_per_block + local_row;
    const int  row       = group_row0 + row_in_g;
    const bool valid_row = row_in_g < m_g;
    const bool valid_vec = valid_row && (tid < nvecs_per_row);

    if (!valid_row) return;

    const int     local_row_in_group = row_in_g;
    const int64_t sf_base            = scale_offsets[group_id];

    constexpr float inv_scales_max = (NumCandidates > 1) ? (1.f / 256.f) : (1.f / 448.f);
    constexpr float inv_six        = 1.f / 6.f;
    float group_amax = amax_ptr[group_id];
    float scale = (group_amax == 0.f) ? 1.f : group_amax * inv_scales_max * (scale_override * inv_six);

    // Write global_scale once per group: only the first CTA's first thread.
    if (local_block == 0 && local_row == 0 && tid == 0)
        global_scale_ptr[group_id] = scale;

    // Load 8 bf16 values.
    bf16x8 x{};
    nv_bfloat16 local_abs_max = __float2bfloat16(0.f);
    if (valid_vec) {
        x = bf16x8::load(x_ptr + 8 * (row * nvecs_per_row + tid));
        local_abs_max = vecReduceAbsMax(x);
    }

    // Per-group-of-16 abs-max (must be called uniformly by all warp threads).
    nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
    float full_abs_max = static_cast<float>(__hmax(local_abs_max, other_abs_max));

    if (!valid_vec) return;

    // Quantize.
    QuantResultGP res;
    if constexpr (NumCandidates == 1) {
        res = quantize_tg(full_abs_max, scale_override * inv_six, scale, x);
    } else {
        float candidates[2] = {6.f, 4.f};
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

    // Store FP8 scale: per-group swizzled OR plain row-major (compile-time switch).
    if (tid % 2 == 0) {
        const int col = tid / 2;
        int64_t tgt;
        if constexpr (Swizzle) {
            const int numKTiles = cols_div16 / 4;
            tgt = cvt_quant_to_fp4_get_sf_out_offset(local_row_in_group, col, numKTiles);
        } else {
            // Row-major layout: [rows_padded, cols_div16].
            tgt = static_cast<int64_t>(local_row_in_group) * cols_div16 + col;
        }
        scale_ptr[sf_base + tgt] = res.fp8s;
    }
}

// ---- Launch helper ----------------------------------------------------------
static void launch_fp4_group_tensor_global(
    int num_candidates,
    __nv_fp4x4_e2m1*   y_ptr,
    __nv_fp8_e4m3*     scale_ptr,
    float*             global_scale_ptr,
    const nv_bfloat16* x_ptr,
    const int32_t*     row_offsets_gpu,
    const int64_t*     scale_offsets_gpu,
    const float*       amax_ptr,          // [num_groups], caller-computed
    const int32_t*     row_offsets_host,  // CPU mirror for CTA offset construction
    float  scale_override,
    long   cols,
    int    num_groups,
    bool   swizzle,
    cudaStream_t stream)
{
    const int nvecs_per_row  = static_cast<int>(cols / 8);
    const int warps_per_row  = (nvecs_per_row + 31) / 32;
    const int rows_per_block = max(1, 1024 / (warps_per_row * 32));
    const int block_dim      = rows_per_block * warps_per_row * 32;
    const int cdiv16         = static_cast<int>(cols / 16);

    // Build group_cta_offsets on the host (O(G) integer arithmetic).
    std::vector<int32_t> h_cta_offsets(static_cast<size_t>(num_groups) + 1);
    h_cta_offsets[0] = 0;
    for (int g = 0; g < num_groups; ++g) {
        int m_g = row_offsets_host[g + 1] - row_offsets_host[g];
        h_cta_offsets[g + 1] = h_cta_offsets[g] + (m_g + rows_per_block - 1) / rows_per_block;
    }
    const int total_ctas = h_cta_offsets[num_groups];

    // Upload group_cta_offsets to GPU (async, same stream).
    auto cta_offsets_t = torch::empty(
        {num_groups + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpyAsync(
        cta_offsets_t.data_ptr<int32_t>(),
        h_cta_offsets.data(),
        static_cast<size_t>(num_groups + 1) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    const int32_t* cta_offsets_ptr = cta_offsets_t.data_ptr<int32_t>();

    #define LAUNCH_TG(NC, SW)                                                          \
        fp4_group_tensor_global_kernel<NC, SW><<<total_ctas, block_dim, 0, stream>>>(  \
            y_ptr, scale_ptr, global_scale_ptr, x_ptr,                                 \
            cta_offsets_ptr, row_offsets_gpu, scale_offsets_gpu,                       \
            amax_ptr, scale_override, cdiv16, num_groups,                              \
            nvecs_per_row, warps_per_row, rows_per_block)

    if (num_candidates == 2) {
        if (swizzle) { LAUNCH_TG(2, true);  }
        else         { LAUNCH_TG(2, false); }
    } else {
        if (swizzle) { LAUNCH_TG(1, true);  }
        else         { LAUNCH_TG(1, false); }
    }
    #undef LAUNCH_TG
    CUDA_CHECK(cudaGetLastError());
}

void rtn_fp4_group_tensor_global_binding(
    at::Tensor&       out,
    at::Tensor&       scales,
    at::Tensor&       global_scale,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& scale_offsets,
    const at::Tensor& amax,
    float             scale_override,
    bool              swizzle)
{
    TORCH_CHECK(inp.dtype()      == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                     "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),               "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                    "inp must be 2D [total_n, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,            "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 1024,           "cols must be <= 8192");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt, "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong,"scale_offsets must be int64");
    TORCH_CHECK(amax.dtype()          == at::kFloat, "amax must be float32");
    const int ngroups = static_cast<int>(row_offsets.numel()) - 1;
    TORCH_CHECK(global_scale.numel() == ngroups,   "global_scale must have num_groups elements");
    TORCH_CHECK(amax.numel()         == ngroups,   "amax must have num_groups elements");

    const long cols = inp.size(1);

    at::Tensor row_offsets_cpu = row_offsets.cpu();
    const int32_t* row_offsets_host = row_offsets_cpu.data_ptr<int32_t>();

    launch_fp4_group_tensor_global(
        1,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        reinterpret_cast<const float*>(amax.data_ptr()),
        row_offsets_host,
        scale_override, cols, ngroups,
        swizzle,
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
    float             scale_override,
    bool              swizzle)
{
    TORCH_CHECK(inp.dtype()      == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                     "inp must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(),               "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                    "inp must be 2D [total_n, cols]");
    TORCH_CHECK(inp.size(1) % 128 == 0,            "cols must be divisible by 128");
    TORCH_CHECK(inp.size(1) / 8 <= 1024,           "cols must be <= 8192");
    TORCH_CHECK(row_offsets.dtype()   == at::kInt, "row_offsets must be int32");
    TORCH_CHECK(scale_offsets.dtype() == at::kLong,"scale_offsets must be int64");
    TORCH_CHECK(amax.dtype()          == at::kFloat, "amax must be float32");
    const int ngroups = static_cast<int>(row_offsets.numel()) - 1;
    TORCH_CHECK(global_scale.numel() == ngroups,   "global_scale must have num_groups elements");
    TORCH_CHECK(amax.numel()         == ngroups,   "amax must have num_groups elements");

    const long cols = inp.size(1);

    at::Tensor row_offsets_cpu = row_offsets.cpu();
    const int32_t* row_offsets_host = row_offsets_cpu.data_ptr<int32_t>();

    launch_fp4_group_tensor_global(
        2,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr()),
        reinterpret_cast<float*>(global_scale.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(scale_offsets.data_ptr()),
        reinterpret_cast<const float*>(amax.data_ptr()),
        row_offsets_host,
        scale_override, cols, ngroups,
        swizzle,
        at::cuda::getCurrentCUDAStream());
}
