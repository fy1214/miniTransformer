#include <cuda.h>
#include <cuda_runtime.h>

#include "../util/transformer_engine/logging.h"
#include "../util/transformer_engine/common.h"
#include "../util/transformer_engine/transformer_engine.h"

constexpr int MXFP8_BLOCK_SIZE = 32;
constexpr int NVFP4_BLOCK_SIZE = 16;

constexpr __device__ __host__ int TB_DIM = 32;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K = 16;
constexpr __device__ __host__ int N_SF_PER_TD_PER_TILE = 4;

// output is in ~K-major interleaved blocks
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K_I32 = NEW_SF_TILE_DIM_K / 4;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_M_I32 = 32;

at::Tensor swizzle_scaling_factors(
  const at::Tensor& input,
  bool rowwise
);