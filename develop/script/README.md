# Megatron MoE 启动入口

用于对比 Transformer Engine 与 miniTransformer 的训练吞吐、step 时间和 nsys
记录。需要匹配的 Megatron、TE 与 CUDA 环境；选择 miniTransformer backend 的同时
会设置层替换开关和量化配方。

## 脚本

| 文件 | 作用 |
|---|---|
| `moe_training.sh` | 原有 TE NVFP4 启动脚本 |
| `moe_training_e2e.sh` | 多 backend 启动；支持具名别名与 `--dry-run` |
| `training.sh` | 原有 dense 训练示例 |
| `moe_e2e_hooks/sitecustomize.py` | TE per-token recipe 和可选 dispatch 记录 hooks |

## 选择 backend

| Backend | 含义 |
|---|---|
| `te_bf16` | 原生 TE BF16 |
| `te_nvfp4` | 原生 TE NVFP4 block scaling |
| `te_nvfp4_pertoken` | 原生 TE per-token |
| `mini_nvfp4_cursor2_cutlass_v2` | 激活 per-token、权重 per-tensor，CUTLASS V2 |
| `mini_nvfp4_per_token` | NVFP4 per-token 前向、MXFP8 反向 |
| `mini_nvfp4_per_token_w_per_tensor` | per-expert tensor weight amax 消融 |
| `mini_nvfp4_per_token_bf16_bwd` | per-token 前向、专用 BF16 反向 |

选择 BF16 反向时，应区分专用反向算子与 layer bypass。完整选择与覆盖规则见
[配置手册](../docs/zh/recipes.md)。

## 预览、检查和运行

```bash
# 预览不会启动训练，也不会创建日志、缓存或 profiler 目录。
bash develop/script/moe_training_e2e.sh \
  --backend mini_nvfp4_per_token --profile off --dry-run

python -m miniTransformer doctor --recipe nvfp4_per_token

# 默认在 DATA_PATH 为空时使用 mock data。
MEGATRON_ROOT=/root/Megatron-LM TRAIN_ITERS=30 \
  bash develop/script/moe_training_e2e.sh \
  --backend mini_nvfp4_per_token --profile off

# BF16 基线与 profiling 入口
bash develop/script/moe_training_e2e.sh --backend te_bf16 --profile off
bash develop/script/moe_training_e2e.sh --backend mini_nvfp4_per_token --profile nsys

# 逐个 backend 对比；每次正常运行生成独立的 perf_summary.json。
for backend in te_bf16 te_nvfp4 mini_nvfp4_per_token mini_nvfp4_per_token_bf16_bwd; do
  BACKEND="$backend" PROFILE=off TRAIN_ITERS=30 bash develop/script/moe_training_e2e.sh
done
```

常用覆盖：`MEGATRON_ROOT`、`MINITRANSFORMER_ROOT`、`CUDA_VISIBLE_DEVICES`、
`TP/PP/CP/EP`、`SEQ_LENGTH`、`NUM_LAYERS`、`LOG_ROOT`、`MINITE_NVFP4_PATH`。
`PROFILE=off` 关闭 profiler，`PROFILE=nsys` 使用 nsys。

脚本以所选 backend 为准，重置调用方残留的 `MINITE_RECIPE`、`QAT_PARAMS`、
`DISABLE_BACKWARD_QUANT` 和层替换开关，再设置匹配的配方。per-token mini backend
默认使用 `MINITE_NVFP4_PATH=dense`，也可显式选择 `auto` 或 `list`。直接 Python
调用的布局默认值仍为 `auto`。

## 运行产物

正常运行写入 `$LOG_DIR/`：

- `run_meta.json`：backend、并行参数、相关环境变量、具名配方、前后向精度和反向来源。
- `train.log`：Megatron 输出。
- `perf_summary.json`：吞吐、iteration 时间和退出状态。
- `nsys/`：`PROFILE=nsys` 的 profiler 记录。

dry-run 只验证配置与命令组织；`doctor` 只检查依赖。数值正确性和训练稳定性需要
在目标 GPU 上运行测试与训练后确认。
