import torch
from typing import Optional
from transformer.ops import quantization
from transformer.utils import utils


class Int4TiledQuantizeOp(quantization.QuantizeOpBase):
    def __init__(self):
        super().__init__()

    @property
    def supports_allgather(self) -> bool:
        return False
    
    def quantize(
        self,
        x: torch.Tensor,
        qparams: quantization.QParams,
        return_transpose: bool = False,
        reduce_amax: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        input_meta: Optional[quantization.QuantizeInputMetaBase] = None,
    ) -> quantization.QuantizeResult:
        
        assert x.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        assert qparams.quant_tile_shape is not None
        assert len(qparams.quant_tile_shape) == 2

        if qparams.quant_tile_shape[0] == 1:
            # Quantize row-wise
            return self.scale_munger.munge_scale_shapes_for_backend(
                self._quantize_vector_tiling(
                    x,
                    qparams.quant_dtype,
                    tile_len=qparams.quant_tile_shape[1],
                    return_transpose=return_transpose,
                    pow_2_scales=qparams.pow_2_scales,
                    eps=qparams.eps,
                ),
                qparams.quant_tile_shape,
            )
        else:
            assert (
                qparams.quant_tile_shape[0] == qparams.quant_tile_shape[1]
            ), "vectorwise_x_and_g_w_per_block requires weight quantization tiles to be square"
            # Quantize block-wise
            return self.scale_munger.munge_scale_shapes_for_backend(
                self._quantize_square_block_tiling(
                    x,
                    qparams.quant_dtype,
                    tile_len=qparams.quant_tile_shape[0],
                    return_transpose=return_transpose,
                    pow_2_scales=qparams.pow_2_scales,
                    eps=qparams.eps,
                ),
                qparams.quant_tile_shape,
            )

