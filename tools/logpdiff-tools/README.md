# logpdiff-tools

训练（Megatron）/ 推理（SGLang）数值一致性诊断工具。对同一个 MoE 层在
两个框架中逐 stage（layer_input → routing → dispatch → fc1 量化/GEMM →
swiglu → fc2 量化/GEMM → 合并输出 → logprob）采集结构化 tensor 记录，
对齐后计算指标、按阈值判定 PASS/WARN/FAIL，并给出**首个异常 stage**，
用于定位 NVFP4 训练/推理不一致的引入点。

## P0 范围

- 仅支持单卡 **TP=EP=PP=1**（writer 会拒绝其他并行配置；多并行采集规划在 P1）。
- 单层、单 batch 样本采集；payload 一律落盘为 `.pt`（CPU 可加载）。
- 依赖仅为**标准库 + torch**（torch 也是懒加载：不碰 tensor 的元数据操作
  在没有 torch 的机器上也能跑）。
- 数据完整性优先：任何 shape / token 数不一致都显式报错或显式标记，
  **禁止 `min(length)` 式静默截断**。

## 目录与 schema

一次 run 的全部产物在 `runs/<run-id>/` 下（run 目录一经创建绝不覆盖）：

```
runs/<run-id>/
├── config.resolved.json   # 完全解析后的运行配置（defaults < config < CLI 合并结果）
├── manifest.json          # 可复现性元数据：git commit、版本、env、seed、
│                          #   parallel{tp,ep,pp}、force_routing、gpu_name
├── launch_meta.json       # 启动时记录的路径/step/source commits/image/recipe；不做权重强哈希
├── inputs.jsonl           # 输入样本（每行一个 JSON）
├── sglang/
│   ├── index.jsonl        # 每行一条 TensorRecord（见下）
│   └── tensors/*.pt       # payload（仅由本工具链读写）
├── megatron/
│   ├── index.jsonl
│   └── tensors/*.pt
├── metrics.json           # compare 输出：stage 指标、判定、首异常点
├── metrics.csv            # 同上，表格形式
└── report.html            # 自包含 HTML 报告
```

`index.jsonl` 每行是一条 `TensorRecord`，字段：

`run_id` / `framework` / `sample_id` / `layer_id` / `tensor_point` /
`global_rank` / `expert_id` / `token_position` / `route_slot` / `shape` /
`dtype` / `provenance` / `payload_path`

- `tensor_point` 枚举见 `logpdiff/schema.py`（`layer_input`、`router_logits`、
  `router_topk_ids/weights`、`dispatch_input`、`fc1_raw_input/qdata/scales/output`、
  `swiglu_output`、`fc2_*`、`expert_combined_output`、`token_logprobs`）。
- `token_position` / `route_slot` 来自 dispatcher 的 token/置换索引，是对齐的
  主键；`expert_id=None` 的记录自成一组（如 `layer_input`）。

## provenance 语义红线

每条记录必须如实标注 `provenance`：

| provenance | 含义 |
|---|---|
| `actual_kernel` | 被测框架真实前向 kernel 的输出（hook 原样抓取） |
| `replayed_reference` | 用参考实现（Python/minite 重算）在采集到的输入上**重放**得到 |
| `hook_boundary` | 模块 hook 边界抓取，可能含 hook 侧变换 |
| `derived_metric` | 由其他 payload 派生（如逐 token logprob 差），非原始数据 |

**红线：`actual_kernel` 与 `replayed_reference` 永远分开呈现、分开比较。**
重放值是参考重算，不是真实 kernel 输出；把 replay 数据混进 actual 比较会
掩盖真实 kernel 差异（例如 SGLang 融合 kernel 内部的 packed qdata 在 Python
侧不可见，只能重放得到，必须标 `replayed_reference`）。报告的 provenance
legend 把两者的 coverage 分开展示；比较 actual QDQ 与 replay QDQ 时这决定
了结论是"kernel 不一致"还是"与参考实现一致"。

## 统一 CLI

```
python -m logpdiff COMMAND [flags]
```

监控与第三章实验辅助命令：

```text
write-launch-meta  记录启动信息，不递归 hash 模型或 tokenizer
export-routing     从 SGLang index 导出 forced-routing .pt
sweep              对指定 layer 逐个调用现有 logpdiff run
aggregate          合并各 run 的 metrics.csv，并报告 coverage/metadata 漂移
rl-metrics         计算 signed δlogp、numerical ratio 和 clip potential
bootstrap          prompt-cluster bootstrap 与 paired condition difference
```

这些命令全部位于 miniTE/logpdiff；不修改 slime。`sweep` 仍然保持“一层一个
不可覆盖 run directory”，只是负责顺序调度现有 `run` 命令。

公共 flag：

- `--config <json>`：配置文件。合并优先级 **内置 defaults < config 文件 < CLI flag**；
  config 中出现未知键（或未知 recipe 名）直接报错，不做静默忽略。
- `--run-dir <path>`：作用于已有 run 目录（compare/report/追加采集）。
- `--strict`：任一 stage FAIL 时以非零码退出（CI 门禁）；不加时仅出报告，退出码为 0。
- `--plots`：报告内联 SVG 图（expert 误差柱状、token 误差趋势、误差直方图、
  ref-vs-target 散点）；默认关闭，报告不含 `<svg>`。
- `--allow-value-match-fallback`：允许 cosine ≥ 0.999 的值匹配兜底对齐
  （默认关闭；见"已知限制"）。

## B200 全流程（一条命令）

```bash
SHOULD_REPLACE_TE_GROUPLINEAR=1 \
SGLANG_NVFP4_PERTOKEN_SCALE=1 \
QAT_PARAMS=8 \
NVTE_NVFP4_DISABLE_RHT=0 \
NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 \
NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
CUDA_VISIBLE_DEVICES=0 \
python -m logpdiff run \
    --model-path /path/to/Qwen3-30B-A3B \
    --ref-load /path/to/Qwen3-30B-A3B_torch_dist \
    --input-file inputs.jsonl \
    --output-dir runs/ \
    --layer-idx 0 \
    --checkpoint-step 0 \
    --source-repo minite=/root/minite \
    --source-repo megatron=/root/Megatron-LM \
    --source-repo slime=/root/slime \
    --strict
```

`run` 在 collector 启动前自动写 `launch_meta.json`。它记录路径、checkpoint
step、source commit、recipe、image/GPU 标签和相关 env；身份策略是
`record_only_no_recursive_hash`，用于追溯与人工核对，不做权重/tokenizer 强哈希门禁。

必要环境变量：

| 变量 | 作用 |
|---|---|
| `SHOULD_REPLACE_TE_GROUPLINEAR=1` | 用 minite 替换 TE GroupedLinear（QAT 量化路径） |
| `SGLANG_NVFP4_PERTOKEN_SCALE=1` | SGLang 侧 NVFP4 使用 per-token scale |
| `QAT_PARAMS=8` | 选择 NVFP4 cursor2 recipe（legacy 数值索引） |
| `NVTE_NVFP4_DISABLE_RHT=0` | 开启 RHT（Random Hadamard Transform） |
| `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0` | 量化开启随机舍入 |
| `NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1` | 关闭 2D 量化（保持 per-token 1D 路径） |

以上 env 会被写入 `manifest.json` 的 `env` 段（连同 `MINITE_*` /
`NVTE_NVFP4_*` / `SGLANG_NVFP4_*` 前缀变量），保证 run 可复现。

## 分步骤命令

```bash
# 1) 采集 SGLang 侧（新建 run；用 --run-id 固定名字便于后续步骤引用）
SHOULD_REPLACE_TE_GROUPLINEAR=1 SGLANG_NVFP4_PERTOKEN_SCALE=1 QAT_PARAMS=8 \
NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 \
NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 CUDA_VISIBLE_DEVICES=0 \
python -m logpdiff collect-sglang \
    --model-path /path/to/Qwen3-30B-A3B \
    --input-file inputs.jsonl --output-dir runs/ \
    --run-id my-run --layer-idx 0

# 2) 采集 Megatron 侧（加入同一 run，写 runs/my-run/megatron/）
SHOULD_REPLACE_TE_GROUPLINEAR=1 QAT_PARAMS=8 \
NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 \
NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
    -m logpdiff collect-megatron \
    --model-path /path/to/Qwen3-30B-A3B \
    --ref-load /path/to/Qwen3-30B-A3B_torch_dist \
    --run-dir runs/my-run --layer-idx 0

# 3) 对齐 + 指标 + 阈值判定（写 metrics.json / metrics.csv；--strict 供 CI）
python -m logpdiff compare --run-dir runs/my-run --strict

# 4) 渲染 HTML 报告（写 report.html；--plots 加内联 SVG 图）
python -m logpdiff report --run-dir runs/my-run --plots
```

每一步都支持 `--config capture.json`；同一键在 CLI 上再次出现则以 CLI 为准。
forced routing 实验加 `--force-routing`（会记入 manifest，报告顶部出现醒目
banner，routing 相关指标仅描述强制路由而非自然行为）。

## 第三章监控辅助流程

### 1. 导出 frozen routing

自然 routing run 完成后，将它已经捕获的 SGLang top-k 数据转成两个 collector
现有的 forced-routing 输入格式：

```bash
python -m logpdiff export-routing \
    --run-dir runs/g1/layer_0000 \
    --output artifacts/frozen_routing.pt
```

命令要求每个 sample 恰好有一组 `[tokens, top_k]` ids/weights，遇到缺失、重复
或 shape 不一致直接报错。

### 2. 稀疏 layer sweep

配置文件使用现有 `logpdiff run` 的 JSON keys，但不能设置 `run_dir`：

```bash
python -m logpdiff sweep \
    --config configs/g2_nvfp4_forced.json \
    --layers 0,4,8,12,14:19,20,24,47 \
    --runs-root runs/ch3 \
    --condition g2_nvfp4_forced
```

已完整生成 `config.resolved.json/launch_meta.json/manifest.json/metrics.json/report.html` 的 layer
会被跳过；已有但不完整的 run 会停止 sweep，避免覆盖或猜测恢复状态。范围采用
Python stop-exclusive 语义，例如 `14:19` 表示 14–18。

`--strict` 导致的数值 FAIL 不会被误判为采集失败：只要完整 artifacts 已生成，
sweep 会把该 layer 记入 `comparison_failures` 并继续。collector/compare 中断且
artifacts 不完整时才停止。

### 3. 聚合 layer metrics

```bash
python -m logpdiff aggregate \
    --runs-root runs/ch3 \
    --output artifacts/layerwise_metrics.csv
```

输出保留原 `metrics.csv` 的全部列，并增加 `condition/run_dir/launch_recipe/
checkpoint_step`。相邻的 `layerwise_metrics.summary.json` 记录 run 数、prompt
coverage 和 metadata 漂移 warning；warning 不做强制指纹阻断。

### 4. δlogp、ratio 与 clip potential

每个 paired run 已包含两侧 chosen-token logp，不需要新增 logp-only collector：

```bash
python -m logpdiff rl-metrics \
    --run-dir runs/ch3/g1/layer_0000 \
    --clip-epsilon 0.2 \
    --condition g1 \
    --output artifacts/g1_rl_metrics.csv
```

符号固定为：

```text
delta_logp = megatron_learner_logp - sglang_rollout_logp
ratio_num  = exp(delta_logp)
```

clip 判定直接在 log-ratio 空间完成，避免 exponent overflow。逐 token CSV 旁会
生成 `.summary.json`，包含 signed mean/median、|δlogp| P50/P95/P99、ratio
P01/P50/P99 和上下侧 clip fraction。

### 5. Prompt-cluster bootstrap

```bash
python -m logpdiff bootstrap \
    --input g0=artifacts/g0_rl_metrics.csv \
    --input g1=artifacts/g1_rl_metrics.csv \
    --input g3=artifacts/g3_rl_metrics.csv \
    --pair g1:g0 \
    --pair g3:g1 \
    --resamples 2000 \
    --seed 20260825 \
    --output artifacts/bootstrap_summary.json
```

paired comparison 要求两条件 prompt 与 token-position coverage 完全一致，不做
静默交集。输出包含 mean/P95/P99 absolute delta 和 clip fraction 的点估计与
95% cluster-bootstrap CI。

## 产物说明

- `config.resolved.json`：三级合并后的最终配置，排障时先看它确认实际生效值。
- `manifest.json`：`git_commit`、`versions`（python/torch/cuda/transformer_engine/
  sglang/megatron/minite）、`env`（白名单内 env）、`seed`、`parallel`、
  `force_routing`、`gpu_name`。
- `sglang/`、`megatron/`：各自 `index.jsonl`（TensorRecord 行）+ `tensors/*.pt`。
  Megatron 侧另保留 `sink_index.jsonl`（minite logpdiff sink 的原始索引）。
  权重类 payload 按 SHA-256 去重（重复记录带 `dedup_ref: true`）。
- `metrics.json` / `metrics.csv`：逐 stage 指标（cosine、对称相对 L2、SQNR、
  分位数绝对误差、routing 匹配率、logprob 超阈值计数等）、PASS/WARN/FAIL 判定、
  对齐统计与首异常点。
- `report.html`：自包含报告：run 头（含对齐方式与 force-routing banner）、
  首异常 stage、stage 表、expert 表（按误差降序）、worst tokens、
  provenance legend（actual/replay 分列）、对齐详情、逐 sample 折叠块。

## 故障诊断入口

1. 打开 `report.html`，看 **First failure** section：首个 FAIL/WARN stage 按
   `PIPELINE_STAGES` 顺序（layer_input → … → logprob）扫描得到，即首个异常点。
2. 若 stage 被标 `INVALID_ALIGNMENT`：该 stage 的 expert 对齐不可靠（shape
   冲突等），数值差异不可信；到 **Expert alignment** section 看逐 expert 原因。
3. 对齐方式 banner：`dispatcher_index`（最可信）→ `hash_exact` →
   `cosine_fallback`（最弱，仅显式开启后出现，报告给出最小接受 cosine）。
4. `layer_input` 就 FAIL ⇒ 两侧输入已不一致，下游所有比较无效，先查输入路径。
5. routing FAIL ⇒ 先看 `boundary_margin_*`：margin 接近 0 说明 top-k 在数值
   噪声下本就会翻转，而不是实现 bug。
6. 需要逐 payload 复查时，用 `logpdiff.writer.assert_internal_payload` 解析
   index 里的 `payload_path` 再 `torch.load`（payload 仅限本工具链自产自用，
   不要指向外部不可信 dump 文件）。

## P0 验收 checklist

- [ ] 1. SGLang self-diff：同框架两次采集 compare，全 stage PASS（位级一致）。
- [ ] 2. Megatron self-diff：同上，全 stage PASS。
- [ ] 3. BF16 对照：两侧 BF16（非 NVFP4）run compare PASS，验证管线与对齐本身无系统性偏差。
- [ ] 4. NVFP4 自然 routing：NVFP4 run compare，routing 及下游 stage 判定符合预期，首异常点可解释。
- [ ] 5. NVFP4 forced routing：`--force-routing` 固定路由后 compare，排除 routing 翻转干扰，隔离 expert 计算差异；报告出现 force-routing banner。
- [ ] 6. actual QDQ 与 replay QDQ 区分：报告中 `actual_kernel` 与 `replayed_reference` coverage 分列，结论只基于同类 provenance 的比较。
- [ ] 7. `report.html` 首异常点人工复核：与 `metrics.json` 的 first_failure 一致，且与该 stage 的 payload 抽查一致。

## 已知限制（P0）

- 仅 TP=EP=PP=1；多 rank 采集（分片 tensor、跨 rank gather）在 P1。
- 对齐主键依赖 dispatcher 索引；无索引侧走 payload 字节 hash + `torch.equal`
  验证（注意 `torch.equal` 视 `NaN != NaN`，含 NaN 的逐字节相同 payload 也
  不会验证通过，保持未匹配状态）。
- 两侧都带 dispatcher 索引的记录**绝不**按值匹配：主键不同即不同 token，
  防止值匹配掩盖 routing bug。
- cosine 值匹配兜底默认关闭（`--allow-value-match-fallback` 开启），阈值
  0.999，最小接受 cosine 会写入结果。
- Megatron 侧不产出 `router_logits`（TopKRouter 不暴露 pre-softmax logits；
  不允许编造内部量）。
- SGLang 侧 packed qdata/scales 为 replay 重算（融合 kernel 不可见），见
  provenance 红线。
- `.pt` payload 是本工具链内部产物，`torch.load` 前必须经
  `assert_internal_payload` 校验路径不逃逸 run 目录。
