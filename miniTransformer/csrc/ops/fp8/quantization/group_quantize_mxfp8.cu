// Grouped MXFP8 quantization kernel (SM100 / Blackwell).
//
// Quantizes a packed BF16 matrix of shape [total_rows, K] (all groups
// concatenated row-wise) into MXFP8 E4M3 data + E8M0 scale factors.
// The scale factor layout matches the CUTLASS SM100 `SfKMajorAtom`
// expected by `gemm_mxfp8_grouped_cutlass.cu`.
//
// Layout recap (SFVecSize=32, Blk_MN=128, Blk_SF=4):
//   SF[mIdx, kIdx]  ->  offset = cvt_quant_to_fp4_get_sf_out_offset(mIdx, kIdx, numKTiles)
//   where  kIdx    = element_index / 32        (k-group index)
//          numKTiles = ceil(K/32 / 4)           (number of 4-kgroup tiles)
//
// The same formula is already implemented in utils.cuh for FP4; it is
// directly reusable here (same SfAtom structure, different SFVecSize).
//
// Performance design (mirrors fp4_group_pertoken_kernel):
//   - Each thread owns ONE bf16x8 vec = 8 contiguous elements -> consecutive
//     lanes touch consecutive 16-byte chunks -> fully coalesced loads/stores.
//   - Values are kept in registers and reused for the quantize pass (no
//     second global read).
//   - A 32-element MXFP8 block = 4 bf16x8 vecs = 4 adjacent lanes; the block
//     abs-max is reduced with two __shfl_xor (mask 1, then mask 2).  No shared
//     memory is needed (each block is independent -- pure block quantization).
//   - Flat grid + grid-stride loop with GPU-side group lookup -> no host sync,
//     no cudaMalloc per call.
//
// Grid / block:
//   nvecs_per_row  = K / 8
//   warps_per_row  = ceil(nvecs_per_row / 32)
//   rows_per_block = max(1, 1024 / (warps_per_row * 32))
//   blockDim       = rows_per_block * warps_per_row * 32
//   gridDim        = min(ceil(total_rows / rows_per_block), MAX_BLOCKS)

#ifndef MINI_TE_NO_BLACKWELL_OPS

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../../../ops.h"
#include "../../../util/quarter/utils.cuh"
#include "../../../util/quarter/vec.cuh"

using bf16x8 = GenericVector<nv_bfloat16, 8>;

// ---------------------------------------------------------------------------
// E8M0 scale computation
// ---------------------------------------------------------------------------

// Returns the E8M0 biased exponent.
// E8M0: unsigned 8-bit, value = 2^(byte - 127).  No NaN/Inf (values 0..254).
// For amax==0 we store 0 (= 2^(-127)) and the quant val is 0 anyway.
__device__ __forceinline__ uint8_t float_to_e8m0(float amax) {
    // OCP MXFP8 / TransformerEngine recipe: normalize amax by the FP8 E4M3
    // max norm (448), then round the resulting exponent UP whenever the
    // mantissa is nonzero.  This maps the block's max element to ~448 (full
    // FP8 dynamic range) instead of [1,2).  Mirrors TE's software fallback in
    // transformer_engine/common/util/ptx.cuh (float_to_e8m0).
    if (amax == 0.0f) return 0u;
    constexpr float max_norm_rcp = 1.0f / 448.0f;   // E4M3 max_norm reciprocal
    const float val = amax * max_norm_rcp;
    const uint32_t bits     = __float_as_uint(val);
    uint32_t       exponent = (bits >> 23) & 0xFF;  // IEEE-754 biased exp
    const uint32_t mantissa = bits & 0x7FFFFFu;
    // Round exponent up on nonzero mantissa, guarding against satfinite
    // (0xFE) overflow and the subnormal boundary (exp==0 && mant<=0x400000).
    if ((mantissa > 0u && exponent != 0xFEu) &&
        !(exponent == 0u && mantissa <= 0x400000u)) {
        ++exponent;
    }
    return static_cast<uint8_t>(exponent);
}

__device__ __forceinline__ float e8m0_to_scale(uint8_t e8m0) {
    // scale = 2^(e8m0 - 127) = ldexpf(1.0f, (int)e8m0 - 127)
    return ldexpf(1.0f, (int)e8m0 - 127);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

static constexpr int MAX_BLOCKS = 1024;

__global__ void mxfp8_group_quantize_kernel(
    uint8_t*            __restrict__ y_ptr,           // [total_rows, K]  FP8 E4M3
    uint8_t*            __restrict__ sf_ptr,          // flat swizzled E8M0 SF buffer
    const nv_bfloat16*  __restrict__ x_ptr,           // [total_rows, K]  BF16 input
    const int32_t*      __restrict__ row_offsets,     // [num_groups+1]  row index per group
    const int64_t*      __restrict__ sf_offsets,      // [num_groups+1]  SF element offset per group
    int    K,                // number of columns
    int    K_groups,         // K / 32
    int    num_groups,
    int    nvecs_per_row,    // K / 8  (one bf16x8 vec per thread-in-row)
    int    warps_per_row,
    int    rows_per_block,
    int    num_k_tiles,      // ceil(K_groups / 4)  -- used in SF offset formula
    int    total_rows,
    int    grid_stride)      // gridDim.x * rows_per_block
{
    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid >> 5;
    const int lane_id     = flat_tid & 31;
    const int local_row   = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    // vec-in-row index (0 .. nvecs_per_row-1); each thread owns one bf16x8.
    const int tid         = warp_in_row * 32 + lane_id;

    // Outer grid-stride loop over row slices.
    for (int base_row = blockIdx.x * rows_per_block;
         base_row < total_rows;
         base_row += grid_stride)
    {
        const int  row       = base_row + local_row;
        const bool valid_row = row < total_rows;
        const bool valid_vec = valid_row && (tid < nvecs_per_row);

        // ---- GPU-side group lookup (linear scan; row_offsets is tiny/cached) ----
        int group_id = 0;
        if (valid_row) {
            while (group_id < num_groups - 1 && row >= row_offsets[group_id + 1])
                ++group_id;
        }
        const int     local_row_in_group = valid_row ? row - row_offsets[group_id] : 0;
        const int64_t sf_base            = valid_row ? sf_offsets[group_id] : 0;

        // ---- Vectorized load of 8 BF16 + per-vec abs-max (single global read) ----
        // One bf16x8 = 16-byte coalesced load; values stay in registers and are
        // reused by the quantize pass below (no second global read).
        bf16x8 x{};
        float  local_amax = 0.0f;
        if (valid_vec) {
            x = bf16x8::load(x_ptr + 8 * ((int64_t)row * nvecs_per_row + tid));
            #pragma unroll
            for (int i = 0; i < 8; i++)
                local_amax = fmaxf(local_amax, fabsf(__bfloat162float(x[i])));
        }

        // ---- 4-lane reduce: one E8M0 scale per 32 elems = 4 adjacent bf16x8 ----
        // mask 1 merges lane pairs; mask 2 merges groups of 4. K%32==0 keeps the
        // valid-lane count a multiple of 4, so each group-of-4 is wholly valid or
        // wholly invalid and never straddles the boundary.
        float blk_amax = local_amax;
        blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 1));
        blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 2));

        if (!valid_vec) continue;

        // ---- E8M0 scale ----
        const uint8_t e8m0_val  = float_to_e8m0(blk_amax);
        const float   scale_fp  = (e8m0_val == 0u) ? 1.0f : e8m0_to_scale(e8m0_val);
        const float   inv_scale = (scale_fp == 0.0f) ? 0.0f : 1.0f / scale_fp;

        // ---- Quantize 8 BF16 -> 8 FP8 E4M3 in registers (no re-read) ----
        constexpr float clamp_v = 448.0f;
        GenericVector<uint8_t, 8> fp8_out;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float scaled = __bfloat162float(x[i]) * inv_scale;
            scaled = fminf(fmaxf(scaled, -clamp_v), clamp_v);
            // Extract the E4M3 storage byte directly: static_cast<uint8_t>(e4m3)
            // would value-convert (via operator float), not reinterpret the bits.
            __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(scaled);
            fp8_out[i] = q.__x;
        }

        // ---- Coalesced vectorized store (8 bytes) via aligned vector store ----
        fp8_out.store(y_ptr + (int64_t)row * K + 8 * tid);

        // ---- Write E8M0 SF to swizzled layout (one thread per 32-elem block) ----
        // kIdx = col = k_group index; mIdx = local_row_in_group.
        if ((tid & 3) == 0) {
            const int     col    = tid >> 2;
            const int64_t sf_off = cvt_quant_to_fp4_get_sf_out_offset(local_row_in_group, col, num_k_tiles);
            sf_ptr[sf_base + sf_off] = e8m0_val;
        }
    } // end grid-stride loop
}

// ---------------------------------------------------------------------------
// Launch helper
// ---------------------------------------------------------------------------

static void launch_mxfp8_group_quantize(
    uint8_t*           y_ptr,
    uint8_t*           sf_ptr,
    const nv_bfloat16* x_ptr,
    int                num_groups,
    int                K,
    int                total_rows,
    const int32_t*     row_offsets_gpu,     // [num_groups+1] on GPU
    const int64_t*     sf_offsets_gpu,      // [num_groups+1] on GPU
    cudaStream_t       stream)
{
    if (total_rows == 0) return;

    const int K_groups      = K / 32;
    const int nvecs_per_row = K / 8;
    const int warps_per_row = (nvecs_per_row + 31) / 32;
    // Target ~1024 threads per block for good occupancy.
    const int rows_per_block = max(1, 1024 / (warps_per_row * 32));
    const int block_dim      = rows_per_block * warps_per_row * 32;
    const int num_k_tiles    = (K_groups + 3) / 4;

    const int natural_grid = (total_rows + rows_per_block - 1) / rows_per_block;
    const int grid_dim     = min(natural_grid, MAX_BLOCKS);
    const int grid_stride  = grid_dim * rows_per_block;

    mxfp8_group_quantize_kernel<<<grid_dim, block_dim, 0, stream>>>(
        y_ptr, sf_ptr, x_ptr,
        row_offsets_gpu, sf_offsets_gpu,
        K, K_groups, num_groups, nvecs_per_row, warps_per_row, rows_per_block,
        num_k_tiles, total_rows, grid_stride);

    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// PyTorch binding
// ---------------------------------------------------------------------------

void mxfp8_grouped_quantize_binding(
    at::Tensor&       out,           // [total_rows, K]  uint8
    at::Tensor&       sf,            // flat swizzled E8M0 buffer  uint8
    const at::Tensor& inp,           // [total_rows, K]  bfloat16
    const at::Tensor& row_offsets,   // [num_groups+1]  int32  GPU
    const at::Tensor& sf_offsets)    // [num_groups+1]  int64  GPU
{
    TORCH_CHECK(inp.dtype()  == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                 "inp must be CUDA");
    TORCH_CHECK(inp.is_contiguous(),           "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                "inp must be 2D [total_rows, K]");
    TORCH_CHECK(inp.size(1) % 32 == 0,         "K must be divisible by 32");
    TORCH_CHECK(row_offsets.dtype()  == at::kInt,  "row_offsets must be int32");
    TORCH_CHECK(sf_offsets.dtype()   == at::kLong, "sf_offsets must be int64");

    const int num_groups = static_cast<int>(row_offsets.numel()) - 1;
    const int total_rows = static_cast<int>(inp.size(0));
    const int K          = static_cast<int>(inp.size(1));

    TORCH_CHECK(out.numel()  == (int64_t)total_rows * K, "out size mismatch");
    TORCH_CHECK(out.dtype()  == at::kByte, "out must be uint8");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inp.device().index());
    launch_mxfp8_group_quantize(
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        reinterpret_cast<uint8_t*>(sf.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        num_groups, K, total_rows,
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(sf_offsets.data_ptr()),
        stream);
}

// ===========================================================================
// Hardware-cvt variant (SM100): full hardware path for E8M0 + FP8 pack.
//
//   E8M0 scale : cvt.rp.satfinite.ue8m0x2.f32   (__nv_cvt_float2_to_e8m0x2)
//   FP8  pack  : cvt.rn.satfinite.e4m3x2.f32    (__nv_cvt_float2_to_fp8x2)
//
// Identical scheduling / memory layout to mxfp8_group_quantize_kernel above;
// only the per-element math is swapped to single-instruction packed cvts.
// May NOT be bit-exact vs the software path at E8M0 rounding boundaries
// (hardware cvt.rp vs hand-tuned float_to_e8m0). Intended as a fast
// inference-grade operator when that small SF-rounding delta is acceptable.
// ===========================================================================

__global__ void mxfp8_group_quantize_hw_kernel(
    uint8_t*            __restrict__ y_ptr,
    uint8_t*            __restrict__ sf_ptr,
    const nv_bfloat16*  __restrict__ x_ptr,
    const int32_t*      __restrict__ row_offsets,
    const int64_t*      __restrict__ sf_offsets,
    int    K,
    int    K_groups,
    int    num_groups,
    int    nvecs_per_row,
    int    warps_per_row,
    int    rows_per_block,
    int    num_k_tiles,
    int    total_rows,
    int    grid_stride)
{
    const int flat_tid    = static_cast<int>(threadIdx.x);
    const int warp_id     = flat_tid >> 5;
    const int lane_id     = flat_tid & 31;
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

        int group_id = 0;
        if (valid_row) {
            while (group_id < num_groups - 1 && row >= row_offsets[group_id + 1])
                ++group_id;
        }
        const int     local_row_in_group = valid_row ? row - row_offsets[group_id] : 0;
        const int64_t sf_base            = valid_row ? sf_offsets[group_id] : 0;

        bf16x8 x{};
        float  local_amax = 0.0f;
        if (valid_vec) {
            x = bf16x8::load(x_ptr + 8 * ((int64_t)row * nvecs_per_row + tid));
            #pragma unroll
            for (int i = 0; i < 8; i++)
                local_amax = fmaxf(local_amax, fabsf(__bfloat162float(x[i])));
        }

        float blk_amax = local_amax;
        blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 1));
        blk_amax = fmaxf(blk_amax, __shfl_xor_sync(0xffffffff, blk_amax, 2));

        if (!valid_vec) continue;

        // ---- E8M0 scale via hardware cvt.rp.satfinite.ue8m0x2 ----
        // Feed amax/448 so the chosen 2^e maps the block max to E4M3 max_norm.
        // cvt.rp (round toward +inf) == smallest 2^e >= amax/448, matching the
        // OCP MX scale recipe. Low byte holds the E8M0 for x.x.
        constexpr float max_norm_rcp = 1.0f / 448.0f;
        const float norm_amax = blk_amax * max_norm_rcp;
        const __nv_fp8x2_storage_t e8_pair =
            __nv_cvt_float2_to_e8m0x2(make_float2(norm_amax, norm_amax),
                                      __NV_SATFINITE, cudaRoundPosInf);
        const uint8_t e8m0_val  = static_cast<uint8_t>(e8_pair & 0xFFu);
        const float   scale_fp  = (e8m0_val == 0u) ? 1.0f : e8m0_to_scale(e8m0_val);
        const float   inv_scale = (scale_fp == 0.0f) ? 0.0f : 1.0f / scale_fp;

        // ---- Quantize 8 BF16 -> 8 FP8 E4M3 via 4x packed cvt.rn.satfinite.e4m3x2 ----
        // satfinite handles the 448 clamp in hardware (no manual fminf/fmaxf).
        GenericVector<uint8_t, 8> fp8_out;
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float2 s2 = make_float2(__bfloat162float(x[i])     * inv_scale,
                                          __bfloat162float(x[i + 1]) * inv_scale);
            const __nv_fp8x2_storage_t p =
                __nv_cvt_float2_to_fp8x2(s2, __NV_SATFINITE, __NV_E4M3);
            fp8_out[i]     = static_cast<uint8_t>(p & 0xFFu);  // low byte = s2.x
            fp8_out[i + 1] = static_cast<uint8_t>(p >> 8);     // high byte = s2.y
        }
        fp8_out.store(y_ptr + (int64_t)row * K + 8 * tid);

        if ((tid & 3) == 0) {
            const int     col    = tid >> 2;
            const int64_t sf_off = cvt_quant_to_fp4_get_sf_out_offset(local_row_in_group, col, num_k_tiles);
            sf_ptr[sf_base + sf_off] = e8m0_val;
        }
    }
}

static void launch_mxfp8_group_quantize_hw(
    uint8_t*           y_ptr,
    uint8_t*           sf_ptr,
    const nv_bfloat16* x_ptr,
    int                num_groups,
    int                K,
    int                total_rows,
    const int32_t*     row_offsets_gpu,
    const int64_t*     sf_offsets_gpu,
    cudaStream_t       stream)
{
    if (total_rows == 0) return;

    const int K_groups      = K / 32;
    const int nvecs_per_row = K / 8;
    const int warps_per_row = (nvecs_per_row + 31) / 32;
    const int rows_per_block = max(1, 1024 / (warps_per_row * 32));
    const int block_dim      = rows_per_block * warps_per_row * 32;
    const int num_k_tiles    = (K_groups + 3) / 4;

    const int natural_grid = (total_rows + rows_per_block - 1) / rows_per_block;
    const int grid_dim     = min(natural_grid, MAX_BLOCKS);
    const int grid_stride  = grid_dim * rows_per_block;

    mxfp8_group_quantize_hw_kernel<<<grid_dim, block_dim, 0, stream>>>(
        y_ptr, sf_ptr, x_ptr,
        row_offsets_gpu, sf_offsets_gpu,
        K, K_groups, num_groups, nvecs_per_row, warps_per_row, rows_per_block,
        num_k_tiles, total_rows, grid_stride);

    CUDA_CHECK(cudaGetLastError());
}

void mxfp8_grouped_quantize_hw_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets)
{
    TORCH_CHECK(inp.dtype()  == at::kBFloat16, "inp must be bfloat16");
    TORCH_CHECK(inp.is_cuda(),                 "inp must be CUDA");
    TORCH_CHECK(inp.is_contiguous(),           "inp must be contiguous");
    TORCH_CHECK(inp.dim() == 2,                "inp must be 2D [total_rows, K]");
    TORCH_CHECK(inp.size(1) % 32 == 0,         "K must be divisible by 32");
    TORCH_CHECK(row_offsets.dtype()  == at::kInt,  "row_offsets must be int32");
    TORCH_CHECK(sf_offsets.dtype()   == at::kLong, "sf_offsets must be int64");

    const int num_groups = static_cast<int>(row_offsets.numel()) - 1;
    const int total_rows = static_cast<int>(inp.size(0));
    const int K          = static_cast<int>(inp.size(1));

    TORCH_CHECK(out.numel()  == (int64_t)total_rows * K, "out size mismatch");
    TORCH_CHECK(out.dtype()  == at::kByte, "out must be uint8");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(inp.device().index());
    launch_mxfp8_group_quantize_hw(
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        reinterpret_cast<uint8_t*>(sf.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(inp.data_ptr()),
        num_groups, K, total_rows,
        reinterpret_cast<const int32_t*>(row_offsets.data_ptr()),
        reinterpret_cast<const int64_t*>(sf_offsets.data_ptr()),
        stream);
}

#else  // MINI_TE_NO_BLACKWELL_OPS — stubs for non-Blackwell builds

#include "../../../ops.h"
#include "../../../util/transformer_engine/logging.h"

void mxfp8_grouped_quantize_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets) {
    NVTE_ERROR("mxfp8_grouped_quantize_binding requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

void mxfp8_grouped_quantize_hw_binding(
    at::Tensor&       out,
    at::Tensor&       sf,
    const at::Tensor& inp,
    const at::Tensor& row_offsets,
    const at::Tensor& sf_offsets) {
    NVTE_ERROR("mxfp8_grouped_quantize_hw_binding requires Blackwell+ (SM100). Rebuild on SM100 hardware or set MINI_TE_FORCE_BLACKWELL_OPS=1.");
}

#endif  // MINI_TE_NO_BLACKWELL_OPS
