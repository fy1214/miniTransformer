# miniTransformer 配置手册

先选择具名配方，再按需要调整现有开关。核心入口是 `MINITE_RECIPE` 或 Python
的 `get_qlinear_params`；QAT1–17 与原枚举接口继续可用。分组表示配方用途，不代表
新的稳定性、精度或性能保证。

## 选择和兼容规则

```bash
MINITE_RECIPE=nvfp4_per_token python train.py
python -m miniTransformer recipes --all
python -m miniTransformer describe nvfp4_per_token
python -m miniTransformer describe 17 --json
python -m miniTransformer doctor --recipe nvfp4_per_token
```

在已接入的程序中，显式传入层的 `qlinear_params` 优先；否则使用环境变量：

- 只设置 `MINITE_RECIPE`：按具名配方选择。
- 只设置 `QAT_PARAMS`：保留旧整数编号的含义。
- 同时设置：必须指向同一配方，否则初始化报错，提示取消一个或改为一致。
- 都未设置或为空：保留 QAT6 reference 回退并提示；新示例始终显式选配方。

所有原 `QuantizeRecipe.value` 都是有效名称，例如
`fp4_subchannel_scaling_pertoken` 与 `nvfp4_per_token` 指向同一配方。
名称忽略大小写和首尾空白，查询工具也接受整数编号。

```python
from miniTransformer.module.config import get_qlinear_params, QuantizeRecipe

params = get_qlinear_params("nvfp4_per_token")
# 同一配方的兼容写法：每次都新建独立配置和算子实例。
params = get_qlinear_params(15)
params = get_qlinear_params(QuantizeRecipe.FP4_SUB_CHANNEL_PERTOKEN)
# 继续通过已有 qlinear_params 参数传给层。
```

`get_qlinear_params_from_predefined`、`QLinearParams`、自定义算子和
`overrides_for_layer_name` 继续可用。例如，可把 `fc2` 设置为独立的
`get_qlinear_params("unquantized")` 配置。已有按层编号关闭量化的逻辑仍优先于
按层名称的覆盖。具名入口不会修改环境变量或自动替换模型里的层。

## 完整配方对照

以下编号以当前实现为准；当前 QAT16 是权重 amax 消融，专用 BF16 backward 是
QAT17。`BF16_REF` 是额外的具名参考配方，不分配新编号。

### 常用配置

| 名称 | QAT_PARAMS | 说明 |
|---|---:|---|
| `unquantized` | 1 | 关闭量化；实际计算 dtype 由层参数决定。 |
| `nvfp4_block` | 4 | 现有 NVFP4 block scaling 配方，保留 RHT 和缩放开关。 |
| `nvfp4_cursor2_cutlass_v2` | 14 | 激活 per-token、权重 per-expert tensor amax；CUTLASS V2 grouped 前向。 |
| `nvfp4_per_token` | 15 | 首页示例：NVFP4 per-token 前向、TE 权重缩放、MXFP8 反向。 |
| `nvfp4_per_token_bf16_bwd` | 17 | NVFP4 per-token 前向，专用 TE-style BF16 反向算子。 |

### 参考实现

| 名称 | QAT_PARAMS | 说明 |
|---|---:|---|
| `bf16_ref` | — | BF16 reference 算子，与关闭量化不同；没有历史 QAT 编号。 |
| `fp4_subchannel_scaling_ref` | 3 | NVFP4 block scaling 参考实现。 |
| `fp4_subchannel_scaling_cursor2_ref` | 6 | Cursor2 reference；未设置选择参数时保留的历史回退。 |

### 实验配置

| 名称 | QAT_PARAMS | 说明 |
|---|---:|---|
| `int4_subchannel_scaling` | 2 | INT4 fake quantization；INT4_FAKE_GROUP_SIZE 默认 16。 |
| `fp4_subchannel_scaling_quartet` | 5 | Quartet NVFP4 配方，保留原有旋转和缩放设置。 |
| `fp4_subchannel_scaling_cursor2` | 7 | Cursor2 路径：激活 per-token、权重 tensor-global amax。 |
| `fp4_subchannel_scaling_cursor2_cutlass` | 8 | 较早的 Cursor2 CUTLASS 路径，供已有实验复用。 |
| `fp4_subchannel_scaling_w4a16` | 9 | 现有 W4A16 前向路径，激活保持高精度。 |
| `fp4_qdq_block` | 10 | NVFP4 block quantize/dequantize 前向实验。 |
| `fp4_qdq_pertoken` | 11 | NVFP4 per-token quantize/dequantize 前向实验。 |
| `fp4_subchannel_scaling_cursor2_cutlass_bwd` | 12 | Cursor2 CUTLASS 前向，MXFP8 CUTLASS dgrad 实验。 |
| `int4_marlin_forward` | 13 | Packed INT4 Marlin 前向；GroupedLinear 需要 DISABLE_BACKWARD_QUANT=1。 |
| `nvfp4_per_token_w_per_tensor` | 16 | QAT15 激活路径 + per-expert tensor weight amax；消融配方。 |

## 反向精度、布局和现有开关

| 配置 | 实际路径 |
|---|---|
| `nvfp4_per_token`，无反向覆盖 | NVFP4 per-token 前向，MXFP8 反向 |
| `nvfp4_per_token` + `DISABLE_BACKWARD_QUANT=1` | 层级 BF16 bypass；兼容 `qat15_disable_bwd` |
| `nvfp4_per_token_bf16_bwd`，无反向覆盖 | 专用 BF16 backward 算子 |
| QAT17 + `DISABLE_BACKWARD_QUANT=1` | 环境变量继续触发层级 bypass；要使用专用算子应取消这个变量 |

查询和运行摘要记录 `backward` 与 `backward_source`，两种 BF16 路径分别呈现。
显式 Python 配置只覆盖配方选择，已有反向、布局和按层关闭量化开关仍生效。

| 参数 | 作用 |
|---|---|
| `MINITE_NVFP4_PATH=auto/dense/list` | per-token 系列的 TE 分组数据包装；普通调用默认 `auto`，e2e 默认 `dense` |
| `DISABLE_BACKWARD_QUANT=1` | 使用原有 BF16 layer bypass |
| `QAT_KEEP_IN_BF16_LAYER_NUMBERS=0,1` | 指定层关闭量化；实际 dtype 由层参数决定 |
| `NVTE_NVFP4_DISABLE_RHT=1` | 关闭支持此开关的配方的 RHT |
| `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1` | 关闭支持此开关的配方的随机舍入 |
| `NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1` | 关闭支持此开关的配方的 2D quantization |
| `INT4_FAKE_GROUP_SIZE` | INT4 fake 配方分组长度；默认 16，保留原有别名 |
| `INT4_REAL_GROUP_SIZE` / `INT4_REAL_SYMMETRIC` | 原有 Marlin 配置；默认 128 / 1 |
| `SHOULD_REPLACE_TE_GROUPLINEAR=1` | 使用已有 Megatron 接入中的 miniTransformer 层替换 |

QAT15–17 的 RHT 和随机舍入由配方固定关闭，NVTE 开关不会重新启用它们。
`MINITE_NVFP4_PATH` 的 `legacy` 与原有布尔别名仍保留；其他字符串依照旧行为作为
`auto` 处理，查询会提示使用规范值。`describe` 说明请求的布局，`doctor` 检查
当前 TE 能否提供该布局；`auto` 根据可用的 dense/list 接口选择。

`recipes` 和 `describe` 只依赖标准库，可以在仓库根目录直接运行。
`doctor` 返回 0 表示依赖检查通过，1 表示缺少依赖或配置条件，2 表示入参错误。
它检查可见 GPU、TE 接口和所选算子的导入，不运行 GEMM，也不保证训练正确性。

## 启动和验证

```bash
bash develop/script/moe_training_e2e.sh \
  --backend mini_nvfp4_per_token --profile off --dry-run
```

完整启动配置见[脚本说明](../../script/README.md)。脚本以 backend 为准，重置残留
的配方选择，再设置一致的 `MINITE_RECIPE` 和 `QAT_PARAMS`。dry-run 打印命令与
配置后退出，不创建日志或缓存目录；正常运行记录 `run_meta.json`。

在仓库根目录执行不需要 GPU 的兼容检查：

```bash
python -m unittest discover -s tests -p 'test_recipe_configuration.py' -v
```

在匹配的 Blackwell / CUDA / TE 环境中执行实际配置与前后向等价检查：

```bash
pytest tests/python/module/test_recipe_api_equivalence.py -v
python tests/python/module/test_group_linear_exact.py --recipe nvfp4_per_token --tests 1 2
```

GPU 测试覆盖配置实例独立性、QAT4/14/15/16/17 的 GroupedLinear 新旧入口等价性，
同时覆盖 QAT4/15 的 dense Linear，并单独覆盖 QAT15 的 BF16 bypass。完整配置实例检查还需要参考和实验配方的相关依赖。
`test_group_linear_exact.py --qparam` 是历史测试编号，**不是 QAT_PARAMS 编号**：
测试编号 9 对应运行配方 QAT15，测试编号 6 对应运行配方 QAT7；旧测试编号不改变。
新用法优先选择 `--recipe`，两个选项互斥。

本轮本机仅进行了无 GPU 检查。CUDA 编译、GPU 数值等价性与完整训练需在目标环境
完成后再报告结果。实现与层接口保持原状，算子导入只延迟到选中配方时执行。
