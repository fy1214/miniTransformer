

/*! \enum NVTEScalingMode
 * \brief Tensor data format.
 */
enum NVTEScalingMode {
  /*! Either an unquantized tensor or an FP8 tensor with per-tensor scaling
   *
   * Not necessary used for delayed tensor scaling. The unintuitive
   * name reflects legacy usage.
   */
  NVTE_DELAYED_TENSOR_SCALING = 0,
  /*! Single scale per block of 32 elements consecutive in either
      rowwise or columnwise direction */
  NVTE_MXFP8_1D_SCALING = 1,
  /*! Tensor is split into NxN quantization tiles or 1xN quantization tiles,
    which each yield a scale. The block_scaling_dim property of the quantizer
    selects the granularity.
   */
  NVTE_BLOCK_SCALING_1D = 2,
  NVTE_BLOCK_SCALING_2D = 3,
  /*! Single scale per block of 16 elements consecutive in either
   * rowwise or columnwise direction */
  NVTE_NVFP4_1D_SCALING = 4,
  NVTE_INVALID_SCALING = 100
};