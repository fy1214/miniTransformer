# miniTransformer

A esay transformer_engine linear framework to use quantization in model training.

## Environment Setup

How to use:

1、use Slime environment, such as slimerl/slime:v0.2.3

```bash
# 拉取最新镜像
docker pull slimerl/slime:v0.2.3

# 启动容器
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:v0.2.3 /bin/bash
```

2、clone the repo with submodules:

```bash
cd /root
git clone --recurse-submodules https://github.com/fy1214/miniTransformer.git
```

Or, if already cloned without submodules:

```bash
git submodule update --init --recursive
```

2、install the package and all dependencies:

```bash
cd miniTransformer

# 安装依赖 + 编译C++扩展
pip install -e . --no-build-isolation
```

This automatically installs all required dependencies and registers the `miniTransformer` package — no manual `export PYTHONPATH` needed.

3、replace Megatron Te linear in /root/Megatron-LM/megatron/core/extensions/transformer_engine.py:

```python
# 如果是dense模型，仅需替换Linear层和LayerNormLinear层
from miniTransformer.module.linear import Linear
from miniTransformer.module.layernorm_linear import LayerNormLinear

class TELinear(Linear):
    ...

class TELayerNormColumnParallelLinear(LayerNormLinear):
    ...
        
# 如果是moe模型，则替换GroupedLinear层

from miniTransformer.module.group_linear import GroupedLinear

class TEGroupedLinear(GroupedLinear):

```

4、使用nvfp4 branch的slime

```shell

```

5、更改sglang文件


## Training metric

Qwen3-4B in nvfp4_quantize_op_ref metric

![Qwen3-4B在nvfp4_quantize_op_ref下loss曲线](develop/docs/img/image.png)