# miniTransformer

A esay transformer_engine linear framework to use quantization in model training.

How to use:

1、build the kernel lib: `python setup.py install`

2、replace Megatron Te linear in transformer_engine.py:

```python
from miniTransformer.module.linear import Linear
class TELinear(Linear):
```

3、train with slime or other script.