1、纯算子精度测试
[test_nvfp4_pertoken.py](quantize_nvfp4_pertoken/test_nvfp4_pertoken.py)：
测试QAT_PARAM=6中的quant算子，即quantize_nvfp4_pertoken 和 quantize_nvfp4_pertoken_swizzled

对比内容：
1、与pytorch golden比较：test_pertoken_rowmajor_vs_golden_rtn
2、与two-pass比较：test_fused_linear_bit_exact

合成输入 pytest 结果（B300，sm_100）：
- test_pertoken_rowmajor_vs_golden_rtn   : 13/13 PASSED（含 cols>8192 two-pass 路径）
- test_pertoken_rowmajor_vs_golden_foursix: 13/13 PASSED
- test_fused_linear_vs_golden            : 11/11 PASSED，2 SKIPPED（cols>8192，预期）
- test_twopass_linear_vs_golden          : 13/13 PASSED

---


2、真实输入测试

获取真实数据方法：加入环境变量，即可启动miniTransformer的profiler。
"LINEAR_STATS_STEPS":"1",
"LINEAR_STATS_BWD": "0",
"LINEAR_STATS_CLASS": "grouped",
"LINEAR_STATS_DUMP_DIR": "/tmp/dump"

实际输入测试（/tmp/dump/fc1_step0000_inp.pt）：
脚本：eval_nvfp4_pertoken_real_input.py（依赖 real_input_eval_utils.py）

输入规格：
- shape: [148480, 2048]，bfloat16，MoE fc1 层真实激活
- 动态范围：min=-3.30，max=3.42，abs_mean=0.080，abs_std=0.114
- 行间动态范围均匀：per-token amax 均值 2.08，标准差 0.44

1. Bit-exact 对比（CUDA kernel vs Python golden，同一真实输入）
   - FP4 data  bit-exact: True
   - FP8 scale bit-exact: True
   - tensor_scale ratio  : min=max=1.000000（与 Python golden amax/(448×6) 一致）
   结论：kernel 量化结果与 Python 参考实现逐 bit 对齐。

2. 量化精度（RTN，反量化后与原始输入对比，基于 fc1_step0000_inp.pt 实测）
   - Effective bitwidth : 3.47（阈值 3.3，通过）
   - Mean MSE           : 1.62e-4
   - Mean cosine sim    : 0.9897
   - Mean row rel-L2    : 9.0%
   注：以上数值依赖输入数据分布，仅供参考。rel-L2 偏高是 FP4 E2M1 格式固有特性
       （行内小值相对于行最大值粒度粗），effective bitwidth 是正确的整体精度指标。

3. global amax vs per-token amax 对比（Python ref，基于 fc1_step0000_inp.pt 实测）
   | 方法            | Eff-bits | Mean cosine | Mean MSE  |
   |-----------------|----------|-------------|-----------|
   | Global amax     |  3.4570  |  0.989665   | 1.64e-4   |
   | Per-token amax  |  3.4658  |  0.989705   | 1.62e-4   |
   | 差值             | +0.009   | +4.1e-5     | -1.9e-6   |

   注：以上数值依赖输入数据分布，仅供参考。该输入行间动态范围均匀（amax std/mean≈21%），
       per-token 优势有限；行间动态范围差异大的输入上 per-token 可能有更明显提升。
       少数行 per-token 更差，来自 FP8 scale 量化舍入方向差异，非策略缺陷。

4. RTN vs Round-4-or-6（基于 fc1_step0000_inp.pt 实测）
   - RTN eff-bits: 3.4658
   - 4/6 eff-bits: 3.5198，提升 +0.054 bit
   注：以上数值依赖输入数据分布，仅供参考。


公共工具：real_input_eval_utils.py
  - load_real_input()   — 加载 dump，flatten 到 2D
  - run_quant_eval()    — 主入口，接收 {label: fn}，输出指标表 + per-row 胜负
  - compute_metrics()   — 返回 eff-bits / MSE / cosine / rel-L2 字典
  - print_metrics_table() / print_row_comparison() — 格式化输出
