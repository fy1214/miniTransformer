# grouped_cutlass_gemm_v2 精度验证

脚本目录：`tests/python/nvfp4/grouped_cutlass_gemm/`

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_grouped_cutlass_gemm_v2_vs_golden.py` | pytest 精度验证：合成数据，kernel vs Python golden |
| `eval_grouped_cutlass_gemm_v2_real_input.py` | 真实数据评测：fc1 step0000 真实激活 + 权重 |

---

## 1. Python Golden 设计

`grouped_cutlass_gemm_v2` 的计算等价于：

```
A_float[i, j] = fp4[i, j]  ×  sf[i, j//16]  ×  act_ts[i]
B_float[g, j] = fp4[g, j]  ×  sf[g, j//16]  ×  wgt_ts[g]
out_g = A_float @ B_float.T          # float32 GEMM
```

其中：
- `sf` 为 tcgen05 swizzled FP8 scale，经 `_unswizzle_scale(flat, m_pad, k)` 还原为 `[m, k//16]`，`m_pad` 需向上对齐到 128 的倍数。
- `act_ts[i]` = per-row global decode scale（= `amax_row / (448×6)`），可携带 per-token 随机因子。
- `wgt_ts[g]` = per-expert global decode scale（= `amax_expert / (448×6)`）。

标量 global scale 可以提到内积之外，因此 dequantize-then-GEMM 与 kernel 的 fused 计算数学等价。

---

## 2. 合成数据测试（pytest）

### 运行方式（SM100 必须）

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
cd /root/miniTransformer
python -m pytest tests/python/nvfp4/grouped_cutlass_gemm/test_grouped_cutlass_gemm_v2_vs_golden.py -v -s
```

### 测试用例

| id | m_splits | n | k |
|----|----------|---|---|
| single_small | [128] | 128 | 128 |
| two_groups | [128, 256] | 256 | 256 |
| three_groups | [128, 128, 128] | 512 | 256 |
| varied_M | [256, 512] | 256 | 512 |
| deepgemm_shape | [128] | 7168 | 4096 |
| moe_8experts_fc1 | [128]×8 | 4096 | 7168 |

seed=[42, 7]，共 12 个参数化用例 + 5 seed sweep，合计 23 个测试。

### 结果（2026-05-30，B300 / SM100）

```
23 passed in 15.49s
```

| 对比 | max_abs | norm_rmse | cosine |
|------|---------|-----------|--------|
| kernel vs Python golden | ≤ 0.0625 | ~0.0033 | ~0.999995 |
| kernel vs FP32 ground truth | — | ~0.13 | ~0.991 |

**kernel vs golden 阈值**：`max_abs < 0.15`，`mean_abs < 0.03`，`norm_rmse < 0.10`，`cosine > 0.999`。

**max_abs = 0.0625（= 2⁻⁴）**：出现在大 shape（k=4096/7168）。BF16 在该量级下一个 ULP = 2⁻⁴，是 CUTLASS 硬件 MMA 路径与 Python float32 路径在大 k 时积累的正常舍入差，非 bug。

**FP32 ground truth 对比注意**：`act_ts` 已包含 per-token 随机因子 `pts`，因此 fp32 参考必须用 `x_fp32 × pts`（而非原始 `x_fp32`），否则会引入 ~30% 的虚假误差。

---

## 3. 真实数据评测

### 数据采集方法

使用 `miniTransformer/profiler/linear_profiler.py` 中的 `LinearStatsHook` 挂载到模型的 `GroupedLinear`（fc1 层），在真实训练/推理的 forward 过程中自动 dump 张量。

**方式一：环境变量（推荐，无需改代码）**

```bash
export LINEAR_STATS_STEPS=1          # 采集 1 步
export LINEAR_STATS_CLASS=grouped    # 挂载 GroupedLinear（而非 Linear）
export LINEAR_STATS_INSTANCE=0       # 第 0 个 GroupedLinear 实例（fc1）
export LINEAR_STATS_SKIP_STEPS=0     # 不跳过，第 1 步即采集
export LINEAR_STATS_DUMP_DIR=/tmp/dump
export LINEAR_STATS_BWD=0            # 只采集前向（不需要梯度）
export LINEAR_STATS_RANK=0           # rank 0 采集
# 然后正常启动训练/推理
```

模型的 `GroupedLinear.__init__` 会自动读取这些变量并挂载 hook。

**方式二：手动挂载**

```python
from miniTransformer.profiler.linear_profiler import LinearStatsHook

hook = LinearStatsHook(
    label="fc1",
    skip_steps=0,
    num_steps=1,
    with_backward=False,
    dump_dir="/tmp/dump",
    rank=0,
)
hook.attach(model.moe.fc1)   # 替换为实际模块路径
```

**dump 产物**（label="fc1"，step=0）：

| 文件 | 内容 |
|------|------|
| `fc1_step0000_inp.pt` | forward 输入激活 |
| `fc1_step0000_weight_NNN.pt` | per-expert 权重（`GroupedLinear` 有多个权重参数） |
| `fc1_step0000_m_splits.json` | 每 expert 的 token 行数列表 |
| `fc1_step0000_output.pt` | forward 输出 |

### 数据来源

容器 `dcf51e50f977`，路径 `/tmp/dump/`：

| 文件 | shape | dtype | 说明 |
|------|-------|-------|------|
| `fc1_step0000_inp.pt` | [148480, 2048] | bf16 | MoE fc1 真实激活 |
| `fc1_step0000_weight_NNN.pt` | [1536, 2048] | bf16 | per-expert 权重（000-063） |
| `fc1_step0000_m_splits.json` | list[64] | — | 每 expert 的行数，sum=148480 |

注：64 个 expert 中有 4 个空 expert（m=0，index 10/15/55/56），eval 脚本已跳过其激活量化。

### 运行方式

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
cd /root/miniTransformer
python3 tests/python/nvfp4/grouped_cutlass_gemm/eval_grouped_cutlass_gemm_v2_real_input.py
```

### 结果（2026-05-30，B300 / SM100）

```
experts=64  n=1536  k=2048  total_m=148480
```

| 对比 | max_abs | norm_rmse | cosine |
|------|---------|-----------|--------|
| kernel vs Python golden | 0.0312 | 0.0032 | 0.999995 |

与合成数据结果一致，kernel 在真实数据上实现正确。

---

## 4. 验证逻辑说明

- **kernel vs Python golden** 是核心正确性验证：两者数学等价，差异仅来自硬件 MMA 路径与 float32 软件路径的舍入顺序。正常范围：norm_rmse < 0.01，cosine > 0.999。超出则说明 kernel 实现有误。
- **max_abs 出现 0.0625（= 2⁻⁴）** 属于正常现象：大 k 时 BF16 在该量级一个 ULP = 2⁻⁴，不是 bug。
- **kernel vs FP32 ground truth** 衡量 FP4 量化本身的精度损失（与 kernel 实现无关），通常 norm_rmse ~0.1–0.2，因输入分布而异。
- **真实数据评测**只关注 kernel vs Python golden，用于确认 kernel 在生产数据分布下无异常行为；具体数值因数据而异，参考合成数据的阈值判断是否正常。
