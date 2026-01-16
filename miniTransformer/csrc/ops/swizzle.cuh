#include <cuda.h>
#include <cuda_runtime.h>

#include "../util/logging.h"
#include "../util/common.h"

constexpr int MXFP8_BLOCK_SIZE = 32;
constexpr int NVFP4_BLOCK_SIZE = 16;

constexpr __device__ __host__ int TB_DIM = 32;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K = 16;
constexpr __device__ __host__ int N_SF_PER_TD_PER_TILE = 4;

// output is in ~K-major interleaved blocks
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K_I32 = NEW_SF_TILE_DIM_K / 4;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_M_I32 = 32;

template <typename LType>
__device__ inline void regs_shuffle(LType* regs_vec) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  if constexpr (N_TILE_PER_TD == 1) return;

  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t tmp[kVectorSize];
  int32_t* ptr = reinterpret_cast<int32_t*>(regs_vec);
#pragma unroll
  for (int i = 0; i < kVectorSize; i++)
    tmp[i % N_TILE_PER_TD * N_SF_PER_TD_PER_TILE + i / N_TILE_PER_TD] = ptr[i];

#pragma unroll
  for (int i = 0; i < kVectorSize; i++) ptr[i] = tmp[i];
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_row_scaling_kernel_impl(const void* input, void* output, const int M,
                                                const int K, const int original_M,
                                                const int original_K, const int bid_x,
                                                const int bid_y, const int grid_dim_x,
                                                const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  // input is in K-major
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M;

  int n_tiles_in_tb = N_TILES_IN_TB;
  const int K_i32 = K / 4;
  if (bid_x == grid_dim_x - 1) {
    n_tiles_in_tb = (K_i32 - 1) % N_TILES_IN_TB + 1;
  }

  bool padding_m = (bid_y == grid_dim_y - 1) && (original_M < M);
  bool padding_k = (bid_x == grid_dim_x - 1) && (original_K < K);

  const int input_offset = bid_y * SF_TILE_DIM_M_I32 * K_i32 + bid_x * N_TILES_IN_TB;
  const int* input_i32 = reinterpret_cast<const int*>(input) + input_offset;
  int* output_i32 = reinterpret_cast<int*>(output) + bid_y * SF_TILE_DIM_M_I32 * K_i32 +
                    bid_x * N_TILES_IN_TB * SF_TILE_SIZE_I32;

  extern __shared__ int4 slm_v4i[];

  // load, global -> regs
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < n_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset = (i * TB_DIM + threadIdx.y) * K_i32 + threadIdx.x * N_TILE_PER_TD;
      regs_vec[i] = __ldg(reinterpret_cast<const LType*>(input_i32 + thread_offset));
      if (padding_m || padding_k) {
        // Pad zeros
        for (int j = 0; j < N_TILE_PER_TD * sizeof(int); j++) {
          const int index = (input_offset + thread_offset) * sizeof(int) + j;
          if (index / K >= original_M || index % K >= original_K) {
            reinterpret_cast<uint8_t*>(regs_vec + i)[j] = 0;
          }
        }
      }
    }

    // shuffle regs
    regs_shuffle<LType>(regs_vec);

// store, regs -> shared
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
      /* TODO rotate i */
      slm_v4i[(threadIdx.x * N_TILE_PER_TD + i) * SF_TILE_SIZE_I32 / 4 + threadIdx.y] =
          reinterpret_cast<int4*>(regs_vec)[i];
    }
  }
  __syncthreads();

  // store, shared -> global
  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  __align__(16) int4* output_v4i = reinterpret_cast<int4*>(output_i32);
#pragma unroll
  for (int i = linear_id; i < SF_TILE_SIZE_I32 * n_tiles_in_tb / 4; i += blockDim.x * blockDim.y) {
    output_v4i[i] = slm_v4i[i];
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_row_scaling_kernel(const void* input, void* output, const int M, const int K,
                               const int original_M, const int original_K) {
  swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
}

void swizzle_scaling_factors(const at::Tensor& input_scale, at::Tensor* output_scale, cudaStream_t stream, NVTEScalingMode scaling_mode) {
  // Check scaling mode
  NVTE_CHECK(scaling_mode == NVTE_MXFP8_1D_SCALING || scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Input tensor has invalid scaling mode (", to_string(scaling_mode()), ").");

  // Deduce tensor dims
  int m{0}, k{0};
  m = input_scale.size(0);
  k = input_scale.size(1);

  // Check dims
  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
  NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");

  const dim3 block_size(TB_DIM, TB_DIM);
  const int num_tiles_m = m / SF_TILE_DIM_M;
  const int num_tiles_k = k / SF_TILE_DIM_K;

  // Perform row-wise swizzle. only row-wise swizzle is needed for NVFP4
  int vec_load_size = (num_tiles_k - 1) % 4 + 1;
  /* there is no int3 and misaligned if using int4/int2 */
  if (vec_load_size == 3) vec_load_size = 1;
  int n_tiles_in_tb = TB_DIM * vec_load_size;
  dim3 num_blocks(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m);
  int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

	int original_M{0}, original_K{0};
	void *input_scale_inv_ptr{nullptr}, *output_scale_inv_ptr{nullptr};
	switch (scaling_mode) {
		case NVTE_MXFP8_1D_SCALING: {
			original_M = m;
			original_K = k / MXFP8_BLOCK_SIZE;
			input_scale_inv_ptr = input_scale.data_ptr();
			output_scale_inv_ptr = output_scale->data_ptr();
			break;
		}
		case NVTE_NVFP4_1D_SCALING: {
			original_M = m;
      original_K = k / NVFP4_BLOCK_SIZE;
      input_scale_inv_ptr = input_scale.data_ptr();
      output_scale_inv_ptr = output_scale->data_ptr();
			break;
		}
		default:
		  NVTE_ERROR("Invalid scaling mode");
	}

	switch (vec_load_size) {
		case 4:
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
      swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks, block_size, slm_size, stream>>>(
              input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
      break;
		case 2:
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
      swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks, block_size, slm_size, stream>>>(
              input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
      break;
		case 1:
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
      swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks, block_size, slm_size, stream>>>(
              input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
      break;
		default:
		  NVTE_ERROR("Not valid vec_load_size.");
		break;
	}
	NVTE_CHECK_CUDA(cudaGetLastError());
}