这里提供 Megatron MoE 训练脚本，用于 **e2e 吞吐 / step 时间 / nsys** 对比（TE vs miniTE）。

## 脚本

| 文件 | 作用 |
|------|------|
| `moe_training.sh` | 单 backend 启动 `pretrain_gpt.py`（Qwen3-30B-A3B MoE 形状） |
| `moe_training_bf16.sh` | **只跑 TE BF16** 的便捷入口（默认 `PROFILE=off` + mock） |
| `moe_training_c3_dapo.sh` | **C3 对齐**：DAPO packed + HF tokenizer + `--load` torch_dist BF16，seed `20260913` |
| `moe_training_sweep.sh` | 扫 backend，汇总 `perf_summary.json` → `compare.json` |
| `moe_e2e_hooks/sitecustomize.py` | `TE_FP4_PERTOKEN=1` 时给 Megatron 打上 TE per-token recipe patch |

## Backends

| backend | 含义 |
|---------|------|
| `te_bf16` | TE GroupedLinear BF16 |
| `te_nvfp4` | TE native NVFP4 block scaling |
| `te_nvfp4_pertoken` | TE `NVFP4PerTokenBlockScaling`（需 hooks） |
| `qat14` | miniTE `SHOULD_REPLACE` + `QAT_PARAMS=14` |
| `qat15` | QAT15 + 默认 `MINITE_NVFP4_PATH=dense` |
| `qat15_disable_bwd` | QAT15 + `DISABLE_BACKWARD_QUANT=1` |
| `qat16` | QAT16（NVFP4 fwd + BF16 bwd op） |

## 用法

```bash
# 只跑 TE BF16（推荐冒烟 / 基线）
bash doc/script/moe_training_bf16.sh
TRAIN_ITERS=10 bash doc/script/moe_training_bf16.sh

# 单跑其它 backend
BACKEND=qat15 PROFILE=nsys TRAIN_ITERS=30 \
  bash doc/script/moe_training.sh

bash doc/script/moe_training.sh --backend te_nvfp4_pertoken --profile off

# C3 对齐：DAPO packed + HF tokenizer + 同一 torch_dist BF16 ckpt
# 资产目录默认 /root/c3_assets（见 moe_training_c3_dapo.sh）
bash doc/script/moe_training_c3_dapo.sh
BACKEND=qat15_disable_bwd TRAIN_ITERS=20 bash doc/script/moe_training_c3_dapo.sh

# 扫全表（默认 mock data；有真实数据时设 DATA_PATH / TOKENIZER_PATH）
BACKENDS="te_bf16 te_nvfp4 te_nvfp4_pertoken qat14 qat15 qat15_disable_bwd qat16" \
  PROFILE=nsys TRAIN_ITERS=30 \
  bash doc/script/moe_training_sweep.sh
```

常用覆盖：`MEGATRON_ROOT` `CUDA_VISIBLE_DEVICES` `TP/PP/CP/EP` `SEQ_LENGTH` `NUM_LAYERS` `LOG_ROOT` `MINITE_NVFP4_PATH`。

产物目录：`$LOG_DIR/`

- `train.log` — Megatron 日志（含 throughput / iteration ms）
- `run_meta.json` — backend / 并行 / env
- `perf_summary.json` — 从日志解析的吞吐摘要
- `nsys/` — nsys 输出（`PROFILE=nsys`）
- Megatron `--profile*` chrome/trace（step 窗口可调 `PROFILE_STEP_START/END`）

## 正确性检查（loss / grad_norm）

本脚本主目标是 **性能**。若要看训练不飞，仍按下列指标对比 TE vs miniTE（短训即可）：

1. **loss** 曲线趋势是否一致（quant/gemm 错了会直接偏）
2. **grad norm** 前 30–50 step 趋势
3. **num zeros** 尽量 <100 量级（wgrad accumulate）

示例曲线见同目录历史图：`grad_norm.png` / `num_zeros.png`。
