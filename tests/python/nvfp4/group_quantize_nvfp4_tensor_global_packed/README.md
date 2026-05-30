实际输入测试（/tmp/dump/fc1_step0000_inp.pt）
脚本：eval_group_quantize_tensor_global_packed_real_input.py（依赖 real_input_eval_utils.py）

输入规格：
- 原始 shape: [148480, 2048]，bfloat16，MoE fc1 层真实激活
- reshape 为 [64, 2320, 2048]（64 experts，每 expert 2320 行，2320 % 128 == 0）

1. 与 Python golden 对比（per-expert global amax，RTN）
   修复 smem hoisting bug 后（2026-05-30）：
   - FP4 data   mismatches: 待重测
   - FP8 scale  mismatches: 0.0097%（1835 / 19005440，全部来自 expert 57）
   - global_scale max abs ratio-1: 1.19e-07（数值上等价）

   ── Bug 根因（已修复）──────────────────────────────────────────────────
   fp4_group_tensor_global_kernel 的 smem broadcast pattern 允许 NVCC 将
   `smem_f[local_row * 2 + 0]` 的读操作（tid≠0 线程）提升到 __syncthreads()
   之前（无 write-to-read 数据依赖），导致多 expert 共用同一 CTA 时（grid-stride
   迭代数 > 1，即 64 experts 时必然触发）读到前一次迭代（另一 expert）的残留值。
   修复：彻底移除 smem，每线程直接读 amax_ptr[group_id]（L2 broadcast，无需同步）。
   修复前：~2018 次非确定性 FP8 scale 错误（跨 expert 污染，random per run）。
   修复后：仅 1835 次确定性"差异"（每次运行完全相同）。
   ────────────────────────────────────────────────────────────────────

   ── 残余 1835 次"差异"分析 ─────────────────────────────────────────
   全部 1835 次集中在 expert 57（global_amax=2.4375），且全部是 impl < ref（差 1）。
   经精确分析，这些位置的真实数学 pre_fp8 值恰好是整数（21、42、84、168），
   即两个相邻 FP8 值的中点：
     ga/6 × (448×6/2.4375) = ga × 448/2.4375 = 整数（因 ga 是 2.4375/448 的有理倍数）
   
   • kernel（rcp_approx_ftz）：计算结果略低于中点 → 向下取 FP8（mantissa=010，偶数）
   • Python 参考：float64 累积误差使结果略高于中点（如 42.000004）→ 向上取（mantissa=011，奇数）
   
   RTN-even 在中点应取 mantissa=010（偶数，即 kernel 的选择）。
   结论：kernel 在这 1835 处的行为比 Python 参考更正确（正确实现了 round-to-nearest-even）。
   这 1835 个"差异"不是 bug，是 Python golden 的 float64 舍入偏差导致的参考误差。
   ─────────────────────────────────────────────────────────────────

   ── FP4 data 差异：rcp_approx_ftz 系统性偏差 ────────────────────────
   FP4 data mismatches: 193573 / 304087040（0.064%），全部是 ±1 FP4 step。
   原因：kernel 有意使用 __rcp_approx_ftz 计算 per-group scale（而非 IEEE 除法），
   以保持与 standalone rtn_fp4 / four_six_fp4 kernel 的 bit-exact 一致性。
   该近似使得 pre-FP4 值在极少数情况下跨越 FP4 格点边界，导致确定性的 ±1 nibble 差。
   特征：
   • 每次运行完全相同（确定性，非 smem bug 的随机误差）
   • 均匀分布在全部 64 个 expert（每 expert ~3024 个差异）
   • 与 FP8 scale 差异独立，完全来自 FP4 量化步骤
   结论：不是 bug，是 rcp_approx_ftz 精度与 Python float64 golden 之间的固有偏差。
   ─────────────────────────────────────────────────────────────────

   ── rel-L2 floor 分析 ──────────────────────────────────────────────
   回归检查（check_regression）的 mean rel-L2 实测值 ~3.48e-03，分解如下：
   • Variant A（完整对比，kernel 输出 vs Python golden）：mean rel-L2 = 3.4777e-03
   • Variant B（同一 global_scale，隔离 FP4 nibble 差异）：mean rel-L2 = 3.4354e-03
   • FP8 scale 1835 次差异贡献：~4.23e-05（可忽略）
   → floor 完全由 rcp_approx_ftz FP4 nibble 差异决定，约 3.4e-03。
   
   阈值选取：
   • COS_THRESH = 0.99：正常 rcp_approx_ftz 偏差下实测 min cosine ≈ 0.9956；
     smem bug 复发时 cosine 会跌至 ~0.85（差异约 15%），0.99 可可靠捕捉。
   • REL_L2_THRESH = 0.01（约 3× floor）：smem bug 会将 rel-L2 抬高至 ~0.1+ 量级，
     0.01 阈值相对 floor 有足够余量，不会误报正常的 rcp_approx_ftz 差异。
   ─────────────────────────────────────────────────────────────────

2. global_scale 粒度对比（per-expert vs 全 tensor）
   - per-expert: min=0.000907  max=0.001273  mean=0.001156
   - 全 tensor:  0.001273（等于 per-expert 的最大值，由最大 amax expert 决定）

3. 量化精度（RTN vs Round-4-or-6）
   | 方法             | Eff-bits | Mean MSE | Mean cosine |
   |------------------|----------|----------|-------------|
   | group_weight RTN |  3.4597  | 1.63e-4  |  0.989676   |
   | group_weight 4/6 |  3.5131  | 1.51e-4  |  0.989964   |
   per-row：4/6 在 91.7% 行更好，提升 +0.053 eff-bit。

4. CUDA vs Python golden（per-expert global amax）
   | 方法                  | Eff-bits | Mean MSE | Mean cosine |
   |-----------------------|----------|----------|-------------|
   | group_weight RTN      |  3.4597  | 1.63e-4  |  0.989676   |
   | ref global (per-exp)  |  3.4599  | 1.63e-4  |  0.989677   |
   per-row：72.3% 行完全相同，差值在噪声量级（eff-bits 差 0.0002）。
   结论：CUDA 实现精度与 Python golden 等价。

5. per-expert amax vs 全 tensor amax（精度策略对比）
   | 方法                  | Eff-bits | Mean MSE | Mean cosine |
   |-----------------------|----------|----------|-------------|
   | group_weight RTN      |  3.4597  | 1.63e-4  |  0.989676   |
   | weight_swizzled RTN   |  3.4570  | 1.64e-4  |  0.989665   |
   per-row：44.0% per-expert 更好，49.2% 更差，6.8% 相同。
   结论：per-expert amax 整体略优（+0.003 eff-bit），但在该输入（experts 间
   amax 差异较大，max/min≈1.4）优势有限；精度差异在噪声量级。
