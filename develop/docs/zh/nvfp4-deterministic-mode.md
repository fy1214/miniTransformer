# Qwen3-30B-A3B NVFP4 Deterministic Mode

English: [`nvfp4-deterministic-mode.en.md`](../en/nvfp4-deterministic-mode.md)

本文说明如何构建运行环境、在容器中完成依赖安装，以及如何启动 Qwen3-30B-A3B 的 **NVFP4 deterministic** 训练与对齐验证流程。

本文覆盖的是 slime 的 **NVFP4 deterministic 对齐栈**：使 Megatron 训练前向与 SGLang rollout 的 token log-probability 在数值上对齐。若需要的是通用 NVFP4 QAT / TEGroupedLinear 训练路径，请参阅：

- [`doc/develop/zh/develop.md`](develop.md)
- [`patch/b200/README.md`](../../../patch/b200/README.md)

文中所有路径均以**环境变量**表示，不绑定特定主机或集群布局。请在使用前按本地环境导出相应变量。

---

## 1. 概述

| 项目 | 说明 |
|---|---|
| 模型 | Qwen3-30B-A3B（MoE）；Hugging Face 权重为 ModelOpt NVFP4（expert Linear 量化；attention、router、LayerNorm 通常保持 BF16） |
| 对齐判据 | `train_rollout_logprob_abs_diff` 低于阈值阈值（默认 `MAX_TRAIN_ROLLOUT_DIFF=9.999e-7`） |
| 训练路径 | Megatron + DeepEP + cutlass NVFP4 per-token MoE（前向）；expert 反向通过 BF16 重计算完成 |
| Rollout 路径 | SGLang `modelopt_fp4` + `cutlass` + `SGLANG_NVFP4_PERTOKEN_SCALE=1` + FA4 + deterministic inference |
| 并行约束 | Tensor Parallel = 1、Expert Tensor Parallel = 1，采用纯 Expert Parallel（默认 `NUM_GPUS=8`）。该对齐配置不支持常见生产布局（例如 TP=4、EP=8） |
| 硬件要求 | 推荐 NVIDIA Blackwell（compute capability sm_100 及以上）；依赖 miniTransformer 提供的 `fp4_gemm` 与 FA4 |

该模式**不能**仅通过修改若干环境变量、在任意 slime 镜像上直接启用。运行前须同时满足：

1. 使用包含 NVFP4 deterministic 对齐实现的 slime 代码树（分支名通常为 `slime-deterministic-patch-nvfp4`，或以等价提交为准）；
2. 已在 Megatron-LM 与 SGLang 上应用 deterministic 及 NVFP4 per-token 相关补丁；
3. 容器环境中已安装 miniTransformer，且 `import fp4_gemm` 成功。

---

## 2. 仓库、分支与目录约定

### 2.1 宿主机构建材料

| 组件 | 建议环境变量 | 说明 |
|---|---|---|
| slime | `$SLIME_SRC` | 含 NVFP4 deterministic 对齐实现的工作树 |
| miniTransformer | `$MINITE_SRC` | 提供 `fp4_gemm` 及 NVFP4 cutlass 算子的工作树 |
| 基础镜像 | `$BASE_IMAGE` | 推荐 `slimerl/slime:latest-cu130`（CUDA 13 / Blackwell）；亦可使用布局兼容的 `slimerl/slime:v0.2.3` 后再叠加补丁 |

请将上述变量指向本地检出路径或镜像标签，勿依赖文档中的示例主机路径。

### 2.2 容器内默认布局

以下默认值与官方 slime 镜像布局一致，可通过环境变量覆盖：

```bash
export SLIME_DIR="${SLIME_DIR:-/root/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
export SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
export MINITE_DIR="${MINITE_DIR:-/root/miniTransformer}"
export PATCH_VERSION="${PATCH_VERSION:-latest}"   # 或 v0.5.15.post1
```

### 2.3 权重与数据（须显式设置）

```bash
export HF_MODEL="${HF_MODEL:?set to the Qwen3-30B-A3B-NVFP4 Hugging Face checkpoint directory}"
export PROMPT_DATA="${PROMPT_DATA:?set to the prompt JSONL file (e.g. dapo-math-17k.jsonl)}"
export EVAL_DATA="${EVAL_DATA:-}"
export SAVE="${SAVE:-/tmp/qwen3_nvfp4_det_ckpt}"
export SGLANG_ROOT="${SGLANG_DIR}"
export MEGATRON_ROOT="${MEGATRON_DIR}"
```

---

## 3. 镜像构建与环境准备

提供两种方式。推荐优先采用**方式 A**：在已发布的 slime 基础镜像上安装补丁与 miniTransformer。

### 3.1 方式 A（推荐）：基础镜像 + 容器内安装

#### 3.1.1 拉取基础镜像并创建容器

```bash
export BASE_IMAGE="${BASE_IMAGE:-slimerl/slime:latest-cu130}"
export CONTAINER_NAME="${CONTAINER_NAME:-slime-nvfp4-det}"

docker pull "${BASE_IMAGE}"

docker run -d --gpus all --name "${CONTAINER_NAME}" \
  --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${HF_MODEL}:${HF_MODEL}:ro" \
  -v "${PROMPT_DATA%/*}:${PROMPT_DATA%/*}:ro" \
  -v "${SLIME_SRC}:${SLIME_DIR}" \
  -v "${MINITE_SRC}:${MINITE_DIR}" \
  "${BASE_IMAGE}" sleep infinity

docker exec -it "${CONTAINER_NAME}" bash
```

可将宿主机构建材料挂载进容器，亦可在容器内通过 `git clone` 获取。slime 工作树须包含 NVFP4 deterministic 对齐实现。

#### 3.1.2 应用补丁并安装 miniTransformer

在容器内执行：

```bash
cd "${SLIME_DIR}"
bash scripts/setup-container-deterministic-patches.sh
```

该脚本将：

1. 向 `$MEGATRON_DIR` 应用  
   `docker/patch/$PATCH_VERSION/megatron.patch` 与  
   `megatron-sglang-aligned.patch`（FP32 residual / fused RMSNorm 对齐）；
2. 向 `$SGLANG_DIR` 应用已存在的补丁，包括  
   `sglang.patch`、`sglang-deterministic.patch`、`sglang-qwen3-deterministic.patch`、  
   `sglang-nvfp4-pertoken.patch` 等；
3. 在 `$MINITE_DIR` 执行  
   `pip install -e . --no-deps --no-build-isolation`，  
   并验证 `import fp4_gemm`。

可选参数：

```bash
SKIP_PATCHES=1 bash scripts/setup-container-deterministic-patches.sh   # 仅安装 miniTransformer
SKIP_MINITE=1  bash scripts/setup-container-deterministic-patches.sh   # 仅应用补丁
PATCH_VERSION=v0.5.15.post1 bash scripts/setup-container-deterministic-patches.sh
```

若 SGLang 的 NVFP4 per-token 实现位于独立工作树，可使用：

```bash
export NVFP4_WORKTREE="${NVFP4_WORKTREE:?path to the SGLang NVFP4 per-token checkout}"
bash scripts/setup-sglang-nvfp4-pertoken.sh "${NVFP4_WORKTREE}"
```

#### 3.1.3 固化镜像（可选）

```bash
docker commit "${CONTAINER_NAME}" "${IMAGE_TAG:?e.g. registry.example.com/slime:nvfp4-det}"
docker push "${IMAGE_TAG}"   # 如需分发
```

### 3.2 方式 B：由 slime `docker/Dockerfile` 重建基础镜像

适用于需要重新发布 CUDA 13 基础镜像的场景。

```bash
cd "${SLIME_SRC}"
just release-cu13
```

仅本地构建、不推送仓库的示例：

```bash
cd "${SLIME_SRC}"
docker build -f docker/Dockerfile . \
  --build-arg DEEPEP_CUDA_ARCH_LIST='10.0 10.3' \
  --build-arg SGLANG_IMAGE_TAG=v0.5.15.post1-cu130 \
  --build-arg PATCH_VERSION=latest \
  -t "${IMAGE_TAG:?registry.example.com/slime:cu130-local}"
```

说明：

- Dockerfile 会应用部分 Megatron / SGLang 补丁，但**不保证**已包含完整的 NVFP4 per-token 与 Qwen3 deterministic 组合。重建后仍建议执行  
  `scripts/setup-container-deterministic-patches.sh`，或确认构建流程已纳入 `sglang-nvfp4-pertoken.patch`。
- slime 应用代码须使用含对齐实现的提交或工作树，不宜直接以未打补丁的上游 `main` 作为对齐运行时。

更多基础镜像发布规则见 slime 仓库中的 `docker/README.md` 与 `docker/justfile`。

---

## 4. 运行前验证

进入容器后，建议按下列顺序检查。

### 4.1 代码与数据

```bash
git -C "${SLIME_DIR}" branch --show-current
git -C "${MEGATRON_DIR}" log -1 --oneline
git -C "${SGLANG_DIR}" log -1 --oneline
test -f "${HF_MODEL}/config.json"
test -f "${PROMPT_DATA}"
```

### 4.2 NVFP4 算子栈

```bash
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}:${SGLANG_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"

python3 - <<'PY'
import fp4_gemm
from sglang.srt.layers.moe.cutlass_moe import (
    cutlass_moe_fp4_pertoken,
    cutlass_moe_fp4_pertoken_deepep_ll,
)
from sglang.srt.layers.quantization.fp4_utils import nvfp4_grouped_gemm, nvfp4_quantize_pertoken
from sglang.srt.server_args import ServerArgs
assert hasattr(fp4_gemm, "grouped_cutlass_gemm_v2")
assert callable(cutlass_moe_fp4_pertoken) and callable(cutlass_moe_fp4_pertoken_deepep_ll)
assert callable(nvfp4_grouped_gemm) and callable(nvfp4_quantize_pertoken)
assert "enable_fp32_moe_router" in ServerArgs.__dataclass_fields__
print("NVFP4 pertoken stack OK")
PY
```

### 4.3 Megatron 对齐补丁

```bash
grep -n '_use_sglang_fused_residual_rmsnorm' \
  "${MEGATRON_DIR}/megatron/core/transformer/transformer_layer.py"
```

若能检索到该符号，表明 `megatron-sglang-aligned.patch` 已生效。

### 4.4 关键环境变量

训练脚本会导出完整环境变量集合。以下变量对行为影响较大：

| 变量 | 典型值 | 说明 |
|---|---|---|
| `SGLANG_NVFP4_PERTOKEN_SCALE` | `1` | 启用 per-token activation scale 的 cutlass MoE 路径 |
| `MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS` | `1` | Megatron 使用与 SGLang 对齐的 FP32 residual RMSNorm |
| `QWEN3_MOE_ALIGNED_SPEC` | `1` | 启用 `qwen3_moe_aligned` attention 插件 |
| `SGLANG_DEEPGEMM_BATCH_INVARIANT` | `1` | DeepGEMM batch-invariant 选择 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | `64` | DeepEP low-latency 填充上限；NVFP4 cutlass 对 `[E, max_m]` 执行整块填充 GEMM，取值过大将显著降低吞吐 |
| `SGLANG_KV_CACHE_DTYPE` | `bfloat16` | KV cache 数据类型 |
| `NUM_GPUS` / `CUDA_VISIBLE_DEVICES` | 须一致 | 决定 EP / DP / actor 规模；二者同时设置时必须相等 |
| `MLP_SOCKET_IFNAME` | 网卡名称 | 多机 NCCL / NVSHMEM bootstrap；单机可按需设置 |

---

## 5. 运行流程

### 5.1 对齐门禁（建议首先执行）

使用较短序列与较小 batch，验证训练与 rollout 的 log-probability 对齐：

```bash
cd "${SLIME_DIR}"
ulimit -n 524288

export HF_MODEL PROMPT_DATA
export SGLANG_ROOT="${SGLANG_DIR}"
export MEGATRON_ROOT="${MEGATRON_DIR}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TRAIN_ROLLOUT_DIFF="${MAX_TRAIN_ROLLOUT_DIFF:-9.999e-7}"
export SGLANG_NVFP4_PERTOKEN_SCALE=1
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}:${SGLANG_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# Optional device pinning (must agree with NUM_GPUS):
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NUM_GPUS=4

python3 -m pytest tests/test_qwen3_30b_a3b_nvfp4_deterministic_e2e.py -s -rs \
  --rollout-batch-size "${SLIME_E2E_ROLLOUT_BATCH_SIZE:-8}"
```

成功条件：进程以退出码 0 结束，且日志包含  
`Qwen3-30B-A3B NVFP4 deterministic train/rollout gate PASSED`。

常见失败原因：

| 现象 | 处理建议 |
|---|---|
| `fp4_gemm` 或 `cutlass_moe_fp4_pertoken` 导入失败 | 重新执行第 3、4 节安装与验证步骤 |
| Megatron 缺少 `_use_sglang_fused_residual_rmsnorm` | 重新应用 `megatron-sglang-aligned.patch` |
| GPU 内存不足 | 降低 `NUM_GPUS`，或减小 `SLIME_E2E_ROLLOUT_BATCH_SIZE` / `SGLANG_MEM_FRACTION_STATIC` |

### 5.2 正式训练脚本

入口脚本：

```bash
cd "${SLIME_DIR}"
ulimit -n 524288

export HF_MODEL PROMPT_DATA SAVE
export REF_LOAD="${REF_LOAD:-${HF_MODEL}}"
export LOAD="${LOAD:-${HF_MODEL}}"
export SGLANG_ROOT="${SGLANG_DIR}"
export MEGATRON_ROOT="${MEGATRON_DIR}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
export ENABLE_EVAL="${ENABLE_EVAL:-0}"
export USE_WANDB="${USE_WANDB:-0}"
export SGLANG_NVFP4_PERTOKEN_SCALE=1

bash scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh
```

默认工作负载形状（均可由环境变量覆盖）：

| 参数 | 默认值 |
|---|---|
| `ROLLOUT_BATCH_SIZE` | 32 |
| `N_SAMPLES_PER_PROMPT` | 8 |
| `GLOBAL_BATCH_SIZE` | 256 |
| `ROLLOUT_MAX_RESPONSE_LEN` | 8192 |
| `ROLLOUT_MAX_CONTEXT_LEN` | 16384 |
| Reward | `--rm-type math` |
| SGLang | `modelopt_fp4` + `cutlass` + FA4 + DeepEP low-latency + deterministic inference |

门禁形状（接近 e2e 测试）：

```bash
SLIME_E2E_SHAPE=1 NUM_ROLLOUT=1 ENABLE_EVAL=0 CI_TEST=1 \
  bash scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh
```

`CI_TEST=1` 将启用  
`--ci-train-rollout-logprob-abs-diff-threshold ${MAX_TRAIN_ROLLOUT_DIFF}`。

四 GPU 示例：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
SLIME_E2E_SHAPE=1 NUM_ROLLOUT=1 ENABLE_EVAL=0 CI_TEST=1 \
  bash scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh
```

### 5.3 脚本执行步骤摘要

1. 清理残留的 `sglang` / `ray` 进程；
2. 导出对齐所需环境变量（见第 4.4 节）；
3. 启动 `ray` head 节点；
4. 通过 `ray job submit` 运行 `python3 train.py`，关键参数包括：
   - `--deterministic-mode`
   - `--megatron-cutlass-nvfp4-moe-forward-layers 0..47`
   - `--custom-megatron-before-*-hook-path ...nvfp4_alignment.enable_nvfp4_alignment_*`
   - `--sglang-quantization modelopt_fp4 --sglang-moe-runner-backend cutlass`
   - `--spec slime_plugins.models.qwen3_moe_aligned get_qwen3_moe_aligned_spec`（当 `QWEN3_MOE_ALIGNED_SPEC=1`）

建议关注的指标：

- `train/train_rollout_logprob_abs_diff`：数值对齐；
- `rollout/raw_reward`、entropy、`grad_norm`：训练行为（与对齐门禁相互独立）。

---

## 6. 故障排除

1. **`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 取值过大**  
   NVFP4 cutlass 在 DeepEP low-latency 路径上对 `[E, max_m]` 执行整块填充 GEMM。若从其他 serving 脚本继承过大取值（例如 1024），MoE 计算量将显著增加。正式训练脚本在未设置 `SGLANG_DEEPEP_ALLOW_LARGE_MAX_M=1` 时会将其限制为 64。

2. **Tensor Parallel 不等于 1**  
   对齐配置要求纯 Expert Parallel。请勿按生产环境的 TP=4 布局修改并行参数。

3. **`CUDA_VISIBLE_DEVICES` 与 `NUM_GPUS` 不一致**  
   训练脚本与 e2e 测试将直接报错退出。

4. **安装 miniTransformer 时修改了容器依赖**  
   必须使用 `pip install -e . --no-deps --no-build-isolation`，以免覆盖镜像中的 PyTorch / Transformer Engine 等组件。

5. **与 `patch/b200/setup.sh` 混用**  
   `patch/b200` 面向另一套 NVFP4 训练配方。deterministic 对齐应以  
   `setup-container-deterministic-patches.sh` 及含对齐实现的 slime 工作树为准。

6. **Reward 指标与对齐判据混淆**  
   部分 Base 权重不会生成 `</think>` 标记，导致 `deepscaler` 类 reward 恒为 0。对齐门禁应以 log-probability 差值为准，不应以 reward 作为通过条件。

7. **共享计算节点上的进程清理**  
   正式脚本启动时可能终止已有的 `sglang` / `ray` 进程。在共享节点上运行前，请确认不会影响其他作业；如脚本版本支持，可使用 `RAY_SKIP_GLOBAL_CLEANUP=1` 并自行管理清理流程。

---

## 7. 检查清单

- [ ] `$BASE_IMAGE` 可用，GPU 设备可见  
- [ ] `$SLIME_DIR` 包含 NVFP4 deterministic 对齐实现  
- [ ] 已成功执行 `setup-container-deterministic-patches.sh`（或完成等价的补丁与 miniTransformer 安装）  
- [ ] 第 4.2、4.3 节验证通过  
- [ ] `$HF_MODEL` 与 `$PROMPT_DATA` 存在且可读  
- [ ] 对齐门禁（e2e）通过  
- [ ] 再启动正式训练形状的 `run-qwen3-30B-A3B-nvfp4-deterministic.sh`  

---

## 8. 相关文件（slime 仓库）

| 路径 | 说明 |
|---|---|
| `scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh` | 正式训练 / 门禁形状入口 |
| `tests/test_qwen3_30b_a3b_nvfp4_deterministic_e2e.py` | pytest 对齐门禁 |
| `scripts/setup-container-deterministic-patches.sh` | 容器内应用补丁并安装 miniTransformer |
| `scripts/setup-sglang-nvfp4-pertoken.sh` | 从独立工作树合并 SGLang NVFP4 per-token 支持 |
| `docker/patch/latest/`（及 `v0.5.15.post1/`） | Megatron / SGLang 补丁源 |
| `slime/backends/megatron_utils/alignment/nvfp4_alignment.py` | Megatron NVFP4 前向挂钩 |
| `slime/backends/megatron_utils/alignment/cutlass_nvfp4_moe_forward.py` | cutlass NVFP4 MoE 实现 |
| `slime/backends/megatron_utils/alignment/residual_checkpoint.py` | 在 activation checkpointing 中保留 FP32 residual |
| `docker/README.md`、`docker/justfile` | 基础镜像构建与发布 |

本仓库（miniTransformer）提供 `fp4_gemm` 扩展（含 `nvfp4_quantize_pertoken`、`grouped_cutlass_gemm_v2` 等），通过  
`pip install -e . --no-deps --no-build-isolation` 安装至运行容器。
