# 开发指南

本文说明如何在 **NVIDIA Blackwell（B200，sm_100+）** 上安装、集成并扩展 miniTransformer。当前量化算子路径仅保证在 Blackwell 上可用。

| 语言 | 文档 |
|------|------|
| 中文（本文） | [`doc/develop/zh/develop.md`](./develop.md) |
| English | [`doc/develop/en/develop.md`](../en/develop.md) |

相关文档：

- [仓库 README](../../../README.md)
- [B200 patch 说明](../../../patch/b200/README.md)
- [MoE e2e 脚本说明](../../script/README.md)

---

## 1. 环境与安装

### 1.1 容器镜像

推荐使用仓库内 [`patch/B300/dockerfile`](../../../patch/B300/dockerfile) 构建开发镜像。该 Dockerfile 基于固定 digest 的 SGLang 镜像，并在镜像内安装 Megatron-LM、Transformer Engine（TE fork）、SGLang、slime 与 miniTransformer。构建上下文可为空（无本地 `COPY` 依赖）。

**B200（sm_100，默认开发路径）：**

```bash
cd /path/to/miniTransformer

DOCKER_BUILDKIT=1 docker build -f patch/B300/dockerfile \
  --build-arg FLASHINFER_CUDA_ARCH_LIST=10.0 \
  --build-arg FLASH_ATTN_CUDA_ARCH_LIST=10.0 \
  --build-arg NVTE_CUDA_ARCHS=100 \
  -t fy121415/dev-cu13-minite-b200:latest \
  .

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it fy121415/dev-cu13-minite-b200:latest /bin/bash
```

**B300（sm_103）示例：**

```bash
DOCKER_BUILDKIT=1 docker build -f patch/B300/dockerfile \
  --build-arg FLASHINFER_CUDA_ARCH_LIST=10.3 \
  --build-arg FLASH_ATTN_CUDA_ARCH_LIST=10.3 \
  --build-arg NVTE_CUDA_ARCHS=103 \
  -t fy121415/dev-cu13-minite-b300:latest \
  .
```

默认基础镜像为 `lmsysorg/sglang@sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356`（`v0.5.12-cu130`）。可通过 `--build-arg SGLANG_IMAGE=...` 覆盖；建议使用 digest，避免浮动 tag。更多 build-arg（TE / Megatron / slime / miniTransformer 分支等）见 Dockerfile 头部注释。

镜像内常见路径：

| 组件 | 路径 |
|------|------|
| miniTransformer | `/root/miniTransformer` |
| Megatron-LM | `/root/Megatron-LM` |
| Transformer Engine | `/root/TransformerEngine`（editable） |
| slime | `/root/slime` |
| SGLang | `/sgl-workspace/sglang` |

### 1.2 在镜像内开发 / 重装 miniTransformer

镜像构建时已按 Dockerfile 中的 `MINITRANSFORMER_BRANCH` 安装 editable miniTransformer。若需切换到本地工作区或最新提交：

```bash
cd /root/miniTransformer
# 同步到目标分支后：
pip install -e . --no-build-isolation --no-deps
```

`pip install -e .` 会编译 C++/CUDA 扩展。安装完成后可直接 `import miniTransformer`，通常无需依赖 `PYTHONPATH` 注入包路径。

### 1.3 增量补丁（可选）

由上述 Dockerfile 构建的镜像已包含 TE FP8 disable、以及 patch 过的 Megatron / slime / SGLang 分支，一般**不必**再跑 `patch/b200/setup.sh`。

若基于旧版 `slimerl/slime:v0.2.3` 或其它未打补丁的基础镜像开发，可执行：

```bash
cd /root/miniTransformer
bash patch/b200/setup.sh
```

该脚本幂等，主要修改如下：

| 目标 | 作用 |
|------|------|
| Megatron-LM | MoE `align_size=128`；使 `TEGroupedLinear` 继承 miniTransformer `GroupedLinear` |
| Transformer Engine | 在 `--fp4` 下将 dense Linear 的 `fp8` 置为 `False`，非专家层保持 BF16 |
| slime（`nvfp4` 分支） | 权重更新后强制执行 `process_weights_after_loading` |
| SGLang | 启用 per-token NVFP4 scale 路径（`SGLANG_NVFP4_PERTOKEN_SCALE`） |

详细说明与已知问题见 [patch/b200/README.md](../../../patch/b200/README.md)。

---

## 2. 仓库结构

```
miniTransformer/
├── miniTransformer/
│   ├── module/           # Linear / LayerNormLinear / GroupedLinear
│   ├── quantiztion/      # QuantizeOp 与 QuantizeRecipe（目录名拼写为 quantiztion）
│   ├── ops/              # Python / CUDA / CuTeDSL / Triton 内核封装
│   ├── csrc/             # C++/CUDA 扩展与 pybind
│   ├── distributed/      # 张量并行、AllReduce 及相关通信
│   ├── utils/env.py      # 环境变量解析（QAT、NVTE、DISABLE_BACKWARD_QUANT 等）
│   └── profiler/         # Linear 级 profiling hook
├── patch/b200/           # B200 一键补丁
├── doc/script/           # MoE e2e、nsys、sweep 启动脚本
├── tests/python/         # 单元测试与 fair benchmark
└── tools/                # 诊断工具（如 logpdiff）
```

MoE 专家路径的调用关系：

```
Megatron TEGroupedLinear
  → miniTransformer.GroupedLinear
    → QLinearParams（由 QAT_PARAMS 映射至 QuantizeRecipe）
      → QuantizeOp.group_quantize / grouped_gemm
        → ops/ 与 csrc/ 中的内核实现
```

---

## 3. 训练集成

### 3.1 slime RL（Qwen3-30B-A3B）

参考 slime 仓库中的 `scripts/run-qwen3-30B-A3B.sh`；模型参数定义于 `scripts/models/qwen3-30B-A3B.sh`。

与该训练配置对齐时，请注意：

- 不启用 `--overlap-grad-reduce` 与 `--overlap-param-gather`（Megatron 默认关闭；与 slime 脚本一致）
- 训练权重通过 `--ref-load` / `--load` 指定；rollout 使用 `--hf-checkpoint`

完整 NVFP4 链路的 `PRECISE_ARGS` 与 Ray `env_vars` 配置见 [patch/b200/README.md](../../../patch/b200/README.md)。

### 3.2 Megatron MoE e2e

请使用 `doc/script/moe_training_e2e.sh` 进行多 backend 对比；保留 `moe_training.sh` 作为原始 TE-only 启动脚本，勿在其上叠加 miniTransformer 逻辑。

```bash
# BF16 冒烟
bash develop/script/moe_training_bf16.sh

# 单 backend
BACKEND=qat15 PROFILE=off TRAIN_ITERS=30 \
  bash develop/script/moe_training_e2e.sh

# 多 backend 扫描
BACKENDS="te_bf16 te_nvfp4 te_nvfp4_pertoken qat14 qat15 qat15_disable_bwd qat16 qat17" \
  bash develop/script/moe_training_sweep.sh
```

`moe_training_e2e.sh` 已与 slime Qwen3 配置对齐（含关闭上述两个 DDP overlap 选项）。更多说明见 [doc/script/README.md](../../script/README.md)。

### 3.3 接入方式

| 方式 | 适用场景 | 行为 |
|------|----------|------|
| 永久补丁（`patch/b200`） | slime / 固定工作镜像 | `TEGroupedLinear` 永久继承 miniTransformer `GroupedLinear` |
| 运行时替换（`SHOULD_REPLACE_TE_GROUPLINEAR=1`） | e2e / fair bench | 在原生 TE `GroupedLinear` 与 miniTransformer 实现之间切换，便于对比 `te_*` 与 `qat*` |

e2e 对 `qat14`–`qat17` 设置 `SHOULD_REPLACE_TE_GROUPLINEAR=1` 与对应 `QAT_PARAMS`；对 `te_bf16` / `te_nvfp4*` 则取消该变量。

Dense 模型可将 Megatron 的 `TELinear` / `TELayerNormColumnParallelLinear` 改为继承 miniTransformer 的 `Linear` / `LayerNormLinear`（见仓库 README）。MoE 主路径为 `GroupedLinear`。

---

## 4. 环境变量

### 4.1 `QAT_PARAMS`

`Linear` 与 `GroupedLinear` 在初始化时通过 `get_qlinear_params_from_env_qat_params()`（`miniTransformer.module.config`）解析该变量。未设置时记录 warning，并默认使用索引 **6**（Cursor2 reference）。

当前主线索引与 e2e backend 对应关系：

| `QAT_PARAMS` | Recipe | e2e backend | 说明 |
|--------------|--------|-------------|------|
| 14 | `FP4_SUB_CHANNEL_CURSOR2_CUTLASS_V2` | `qat14` | Cursor2 CUTLASS V2 |
| 15 | `FP4_SUB_CHANNEL_PERTOKEN` | `qat15` | 激活 per-token；默认 `MINITE_NVFP4_PATH=dense` |
| 16 | `FP4_SUB_CHANNEL_PERTOKEN_W_PERTENSOR` | `qat16` | QAT15 激活路径 + 权重 per-expert tensor amax |
| 17 | `FP4_SUB_CHANNEL_PERTOKEN_BF16_BWD` | `qat17` | 前向 per-token；反向 BF16 |

索引 1–13（INT4、quartet、历史 Cursor2、QDQ 等）见 `config.py` 中的 `get_qlinear_params_from_env_qat_params`。

> **注意：** 部分旧文档或脚本使用 `QAT_PARAM`（无 `S`）。运行时仅识别 `QAT_PARAMS`。

### 4.2 `SHOULD_REPLACE_TE_GROUPLINEAR`

| 取值 | 行为 |
|------|------|
| `1` | 使用 miniTransformer `GroupedLinear`，并由 `QAT_PARAMS` 选择 recipe |
| 未设置或其他 | 使用 TE 原生 `GroupedLinear`（可配合 `--fp4-*` 启用 TE NVFP4） |

### 4.3 其他常用变量

| 变量 | 含义 |
|------|------|
| `DISABLE_BACKWARD_QUANT=1` | 反向跳过量化，BF16 直通（e2e：`qat15_disable_bwd`） |
| `MINITE_NVFP4_PATH` | 权重量化布局；QAT15/16 默认 `dense` |
| `NVTE_NVFP4_DISABLE_RHT` | `0` 启用 RHT；`1` 关闭 |
| `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING` | `0` 启用随机舍入；`1` 关闭 |
| `NVTE_NVFP4_DISABLE_2D_QUANTIZATION` | `1` 使用 1D weight scale（对齐 SGLang `modelopt_fp4`） |
| `TE_FP4_PERTOKEN=1` | e2e `te_nvfp4_pertoken`：通过 hooks 将 Megatron recipe 切换为 TE per-token |
| `MOE_DUMP_DISPATCH=1` | 通过 e2e hooks 导出 per-expert token 直方图 |

解析实现位于 `miniTransformer/utils/env.py`。

---

## 5. 添加 QuantizeOp

目标是使新 recipe 可通过 `QAT_PARAMS=N` 被选中，并由 `GroupedLinear` / `Linear` 调用。

### 5.1 实现算子

在 `miniTransformer/quantiztion/` 下新增模块，继承 `QuantizeOpBase`（定义于 `quantization.py`），并实现：

| 方法 | 用途 |
|------|------|
| `quantize(...)` | Dense `Linear` |
| `group_quantize(...)` | MoE `GroupedLinear`（按 `m_splits`） |
| `qgemm(...)` | Dense GEMM |
| `grouped_gemm(...)` | MoE grouped GEMM（MoE 路径必需） |
| `supports_allgather` | 张量并行下是否支持「先量化再 all-gather」；多数 NVFP4 实现为 `False` |

参考实现：

- `nvfp4_quantize_op_pertoken.py`（QAT15）
- `nvfp4_cursor2_op_cutlass_v2.py`（QAT14）

内核实现放在 `ops/` 或 `csrc/`；Python 层负责编排与接口适配。

### 5.2 注册 recipe 与索引

编辑 `miniTransformer/module/config.py`：

1. 在 `QuantizeRecipe` 中新增枚举值；
2. 在 `get_qlinear_params_from_predefined(...)` 中返回包含 `quantize_op=YourOp()` 的 `QLinearParams`；
3. 在 `get_qlinear_params_from_env_qat_params()` 中增加 `qat_params_idx == N` 分支。

如需纳入 e2e，可在 `doc/script/moe_training_e2e.sh` 中增加对应 backend，并设置 `SHOULD_REPLACE_TE_GROUPLINEAR=1` 与 `QAT_PARAMS=N`。

### 5.3 冒烟验证

```bash
cd tests/python/module
python test_group_linear_exact.py --qparam 15 --tests 1 2

QAT_PARAMS=N python test_group_linear_exact.py --tests 1 2
```

随后可按需补充 `tests/python/nvfp4/` 下的精度测试，以及 `doc/script` 中的 e2e 配置。

---

## 6. 测试

以下命令假定当前目录为仓库根目录，已完成 `pip install -e .`，并在 Blackwell GPU 上执行。

### 6.1 NVFP4（pytest）

```bash
pytest tests/python/nvfp4 -m "production and not bench" -v

pytest tests/python/nvfp4/quantize_nvfp4_pertoken/test_nvfp4_pertoken.py -v
```

pytest marker（`production` / `legacy` / `eval` / `bench` / `slow`）说明见 [tests/python/nvfp4/README.md](../../../tests/python/nvfp4/README.md)。

### 6.2 Module

```bash
python tests/python/module/test_group_linear_exact.py
python tests/python/module/test_group_linear_exact.py --qparam 14 --tests 1 2 3

python tests/python/module/test_group_linear_te_vs_mini.py
```

### 6.3 Megatron fair / 低比特

```bash
torchrun --nproc_per_node=4 \
  tests/python/megatron/bench_moe_pertoken_fair.py \
  --backend mini_qat15 --bench-mode fwb --preset qwen3-30b-a3b \
  --tp-size 4 --moe --moe-grouped-gemm --seq-length 4096 --num-layers 1

bash tests/python/megatron/run_fair_bench_te_vs_mini.sh
```

更完整的 e2e 与 nsys 流程见 `doc/script/`。

### 6.4 分布式

见 [tests/python/distributed/README.md](../../../tests/python/distributed/README.md)。

---

## 7. 新环境检查清单

1. 按 §1.1 用 `patch/B300/dockerfile` 构建并启动镜像（B200 默认路径）。
2. 如需本地改代码，在 `/root/miniTransformer` 执行 `pip install -e . --no-build-isolation --no-deps`。
3. 仅在非 Dockerfile 基础镜像上，才执行 `bash patch/b200/setup.sh`。
4. 运行 `pytest tests/python/nvfp4 -m "production and not bench" -q`。
5. 运行 `BACKEND=te_bf16 PROFILE=off TRAIN_ITERS=5 bash doc/script/moe_training_bf16.sh`。
6. （可选）运行 `BACKEND=qat15 ... bash doc/script/moe_training_e2e.sh`。
7. 使用 slime / e2e 时确认未误开 DDP overlap（`--overlap-grad-reduce` / `--overlap-param-gather`）。
