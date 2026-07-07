# B200 NVFP4 全流程工作镜像

针对 `slimerl/slime:v0.2.3` 基础镜像，把 NVFP4 训练 + 推理需要的所有补丁集中
在这里，一键脚本 `setup.sh` 负责打到对应位置。

## 目录结构

```
patch/b200/
├── README.md                       # 当前文件
├── setup.sh                        # 一键安装入口
├── megatron/
│   └── 01-quantization-align-and-grouped-linear.patch
├── transformer-engine/
│   ├── apply_disable_fp8.py        # 幂等 Python patcher (TE 版本变化频繁)
│   └── README.md
├── slime/
│   └── 01-force-post-process-weights.patch
└── sglang/
    └── nvfp4-pertoken-scale-20260611.patch
```

## 镜像构建流程 (容器内)

```bash
# 1. 拉镜像并启动 (主机执行)
docker pull slimerl/slime:v0.2.3
docker run --gpus all -it --name b200-nvfp4 \
    -v /home/admin/mingfa:/root/workspace \
    slimerl/slime:v0.2.3 bash

# 2. (容器内) clone miniTransformer + slime 到 /root
cd /root
git clone <miniTransformer-url> miniTransformer
git clone -b nvfp4 https://github.com/fy1214/slime.git

# 3. 一键安装
cd /root/miniTransformer
bash patch/b200/setup.sh

# 4. (可选) commit 成新镜像
exit
docker commit b200-nvfp4 slimerl/slime:v0.2.3-nvfp4-b200
```

## 各 patch 的作用

| Patch | 目标 | 做了什么 | 为什么 |
|-------|------|----------|--------|
| `megatron/01-...patch` | Megatron-LM | (1) `Fp8Padding/Unpadding` 改 `align_size=128` (2) `TEGroupedLinear` 改继承 `miniTransformer.GroupedLinear` | NVFP4 要求 MoE 专家 group 128 对齐；用 miniTE 的 grouped-linear 替换 TE 默认实现 |
| `transformer-engine/apply_disable_fp8.py` | TE `linear.py` / `layernorm_linear.py` | 在 `_Linear.forward` / `_LayerNormLinear.forward` 解开 `non_tensor_args` 之后写入 `fp8 = False` | `--fp4` 训练默认让所有 Linear 走量化；MoE 之外的层应保持 bf16 |
| `slime/01-...patch` | slime `update_weight_from_tensor.py` | 把第二个 `if (self.quantization_config ...)` 改成 `if True` | rollout 端每次更新权重后必须重新跑 `process_weights_after_loading`（nvfp4 expert-only checkpoint 需要在 sglang engine 上重打包） |
| `sglang/nvfp4-pertoken-scale-*.patch` | sglang | 新增 per-token activation scale 的 cutlass MoE 路径 + 环境开关 (`SGLANG_NVFP4_PERTOKEN_SCALE`) | 把激活量化粒度从 per-tensor 提到 per-token，可消除 ~0.012 的 logits diff (参见 [[nvfp4-pertoken-diff-analysis]]) |

## 训练参数 (供 `slime` 训练脚本使用)

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-quantization modelopt_fp4
   # --sglang-ep-size 2
   # --sglang-moe-runner-backend triton
   # --sglang-moe-a2a-backend deepep
   # --sglang-disable-cuda-graph
)

PRECISE_ARGS=(
   --transformer-impl transformer_engine
   --bf16
   --fp4-format e2m1
   --fp4-recipe nvfp4
   # --fp4-param-gather
)
```

```jsonc
// ray runtime env
"env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/:/root/miniTransformer",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NVTE_NVFP4_DISABLE_RHT": "0",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "0",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",  // 1D scale — 对齐 sglang modelopt_fp4 推理
    "QAT_PARAM": "8"
}
```

环境变量解释：
- `NVTE_NVFP4_DISABLE_RHT=0` — backward 时开 RHT (Random Hadamard Transform)
- `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0` — 量化时开随机舍入
- `NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1` — 权重保持 1D 量化；当前 sglang modelopt_fp4
  推理用的就是 1D 内核，开 2D block-scale 推理会有问题
- `QAT_PARAM=8` — QAT 参数粒度（对应 V2 transpose-quantize 的 QAT8 配方）

## 模型权重准备

NVFP4 expert-only quant 模型由 slime nvfp4 分支的工具生成：

```bash
cd /root/slime  # 已 checkout nvfp4 分支
python tools/convert_hf_to_nvfp4.py \
    --model-dir  /root/workspace/model/hf/Qwen3-30B-A3B \
    --save-dir   /root/workspace/model/hf/Qwen3-30B-A3B-NVFP4
```

## 重新打 patch / 升级

- patch 是幂等的：`setup.sh` 重跑安全
- 单独重打某一组：
  ```bash
  cd /root/Megatron-LM && git apply /root/miniTransformer/patch/b200/megatron/*.patch
  python3 /root/miniTransformer/patch/b200/transformer-engine/apply_disable_fp8.py
  cd /root/slime && git apply /root/miniTransformer/patch/b200/slime/*.patch
  ```
- TE 升级后 patcher 失效：检查 `apply_disable_fp8.py` 里的 `MARKER_RE`，新版本可能
  不再用 `non_tensor_args` 解包

## 已知坑

1. TE patcher 依赖 `non_tensor_args` 解包标记。`slimerl/slime:v0.2.3` 镜像内的 TE
   是这样写的；若 TE 升级到没有此模式的版本，需要手动加 `fp8 = False`。
2. sglang patch 当前是 dist-packages 路径 (`/usr/local/lib/python3.12/.../sglang`)，
   不是 editable install。`setup.sh` 会用 `patch -p2` 兜底。如果你把 sglang
   切到 editable install，把 `SGLANG_DIR` 指过去就行。
3. slime 的 patch 写在 `nvfp4` 分支（fy1214 fork）。如果用主干 slime，
   `update_weight_from_tensor.py` 路径可能不存在，setup.sh 会自动 skip。
