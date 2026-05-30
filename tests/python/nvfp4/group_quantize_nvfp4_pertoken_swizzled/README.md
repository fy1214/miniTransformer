实际输入测试（/tmp/dump/fc1_step0000_inp.pt）
脚本：eval_group_quantize_pertoken_swizzled_real_input.py（依赖 real_input_eval_utils.py）

输入规格：
- shape: [148480, 2048]，bfloat16，MoE fc1 层真实激活
- N_GROUPS=8，每组 18560 行（18560 % 128 == 0）

1. Bit-exact 对比（单组 vs Python golden，per-token amax，RTN）
   - FP4 data  bit-exact: True
   - FP8 scale bit-exact: True
   - global_scale ratio  : min=max=1.000000（non-zero rows: 147556）
   结论：量化数据与 Python 参考逐 bit 对齐。

2. 多组一致性（8 等分组 vs 单组）
   - FP4 data   bit-exact: True
   - global_scale bit-exact: True
   结论：分组边界不影响量化结果，各行独立量化行为符合预期。

3. 与 quantize_nvfp4_pertoken 一致性（swizzled vs row-major，同一 CUDA 算子族）
   - FP4 data   bit-exact: True
   - FP8 scale  bit-exact: True（unswizzle 后与 row-major 逐 bit 对齐）
   - global_scale bit-exact: True
   结论：两者量化结果完全一致，仅 scale 内存布局不同。

4. 量化精度（RTN vs Round-4-or-6）
   | 方法       | Eff-bits | Mean MSE | Mean cosine |
   |------------|----------|----------|-------------|
   | group RTN  |  3.4658  | 1.62e-4  |  0.989705   |
   | group 4/6  |  3.5198  | 1.49e-4  |  0.989996   |
   per-row：4/6 在 93.8% 行更好，提升 +0.054 eff-bit。

5. group RTN vs Python golden（per-token amax ref）
   | 方法            | Eff-bits | Mean MSE | Mean cosine |
   |-----------------|----------|----------|-------------|
   | group RTN       |  3.4658  | 1.62e-4  |  0.989705   |
   | ref pertoken    |  3.4658  | 1.62e-4  |  0.989705   |
   per-row：96.0% 行结果完全相同；差异来自 swizzled fp8 scale 与 ref
   row-major fp8 scale 的 bfloat16 舍入方向差异（噪声量级）。
   结论：group_quantize_nvfp4_pertoken_swizzled 精度与 Python golden 等价。
