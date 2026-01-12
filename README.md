# miniTransformer

A esay transformer_engine linear framework to use quantization in model training.

## Environment Setup

How to use:

1、build the kernel lib: `python setup.py install`

2、replace Megatron Te linear in transformer_engine.py:

```python
from miniTransformer.module.linear import Linear
class TELinear(Linear):
```

3、train with slime or other script.

## Training metric

Qwen3-4B in nvfp4_quantize_op_ref metric

![Qwen3-4B在nvfp4_quantize_op_ref下loss曲线](doc/images/image.png)