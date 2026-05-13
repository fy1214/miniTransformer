# 开发教程

## 容器

注：miniTransformer算子目前只支持BlackWell 系列

如果你想要开发miniTransformer, 你可以使用 `nvcr.io/nvidia/pytorch:25.11-py3` 镜像作为起始镜像.

```shell
# 拉取最新镜像
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# 启动容器
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it nvcr.io/nvidia/pytorch:25.11-py3 /bin/bash
```

## 安装 miniTransformer

miniTransformer目前只需要依赖TransformerEngine，因此你需要在镜像中先安装TE

```shell
# 编译TE
git clone https://github.com/NVIDIA/TransformerEngine.git
git submodule update --init --recursive
NVTE_BUILD_DEBUG=1 pip3 install --no-build-isolation .
```

然后编译miniTransformer

```shell
git submodule update --init
python setup.py install
```


## 流程运行

如果你想将miniTransformer用在训练上，你需要搭配Megatron和一起使用。