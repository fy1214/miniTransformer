# miniTransformer recipe guide

Choose a named recipe, then adjust existing tuning flags as needed. Use
`MINITE_RECIPE` or Python's `get_qlinear_params`; QAT1–17 and existing enum APIs
remain available. Groups describe purpose, not new stability, accuracy, or
performance guarantees.

## Selection and compatibility

```bash
MINITE_RECIPE=nvfp4_per_token python train.py
python -m miniTransformer recipes --all
python -m miniTransformer describe nvfp4_per_token
python -m miniTransformer describe 17 --json
python -m miniTransformer doctor --recipe nvfp4_per_token
```

An explicit layer `qlinear_params` takes priority over recipe environment
selectors. Otherwise:

- `MINITE_RECIPE` alone selects a named recipe.
- `QAT_PARAMS` alone keeps the original integer mapping.
- Both must select the same recipe; conflicts report how to unset or align them.
- Neither set, or both empty: preserve the QAT6 reference fallback with a warning.
  New examples always choose a recipe explicitly.

Every original `QuantizeRecipe.value` remains a valid name. For example,
`fp4_subchannel_scaling_pertoken` is an alias of `nvfp4_per_token`. Names ignore
case and surrounding whitespace; inspection commands also accept integer QAT IDs.

```python
from miniTransformer.module.config import get_qlinear_params, QuantizeRecipe

params = get_qlinear_params("nvfp4_per_token")
# Compatible spellings; every call constructs independent parameters/operators.
params = get_qlinear_params(15)
params = get_qlinear_params(QuantizeRecipe.FP4_SUB_CHANNEL_PERTOKEN)
# Pass through the existing layer qlinear_params argument.
```

`get_qlinear_params_from_predefined`, `QLinearParams`, custom operators, and
`overrides_for_layer_name` remain available. For example, assign a separate
`get_qlinear_params("unquantized")` configuration to `fc2`. Existing per-layer
number exclusions still precede per-layer name overrides. The factory does not
change environment variables or replace model layers.

## Complete recipe mapping

Numbers below reflect this checkout. QAT16 is the weight-amax ablation; the
dedicated BF16 backward recipe is QAT17. `BF16_REF` is an additional named
reference recipe with no new QAT number.

### Common

| Name | QAT_PARAMS | Purpose |
|---|---:|---|
| `unquantized` | 1 | Quantization disabled; computation dtype is controlled by the layer. |
| `nvfp4_block` | 4 | Existing NVFP4 block-scaling recipe; RHT and scaling toggles remain available. |
| `nvfp4_cursor2_cutlass_v2` | 14 | Per-token activations, per-expert tensor weight amax, CUTLASS V2 grouped forward GEMM. |
| `nvfp4_per_token` | 15 | Per-token NVFP4 forward with TE weight scaling and MXFP8 backward. First-use example. |
| `nvfp4_per_token_bf16_bwd` | 17 | Per-token NVFP4 forward with the dedicated TE-style BF16 backward operator. |

### Reference

| Name | QAT_PARAMS | Purpose |
|---|---:|---|
| `bf16_ref` | — | BF16 reference operator; distinct from disabling quantization. |
| `fp4_subchannel_scaling_ref` | 3 | Reference implementation of NVFP4 block scaling. |
| `fp4_subchannel_scaling_cursor2_ref` | 6 | Cursor2 reference path; historical fallback when neither selector is set. |

### Experimental

| Name | QAT_PARAMS | Purpose |
|---|---:|---|
| `int4_subchannel_scaling` | 2 | INT4 fake quantization; group size is set by INT4_FAKE_GROUP_SIZE (default 16). |
| `fp4_subchannel_scaling_quartet` | 5 | Quartet NVFP4 recipe with its existing rotation and scaling settings. |
| `fp4_subchannel_scaling_cursor2` | 7 | Cursor2 path: per-token activations and tensor-global weight amax. |
| `fp4_subchannel_scaling_cursor2_cutlass` | 8 | Earlier Cursor2 CUTLASS path; retained for existing experiments. |
| `fp4_subchannel_scaling_w4a16` | 9 | Existing W4A16 forward path with high-precision activations. |
| `fp4_qdq_block` | 10 | NVFP4 block quantize/dequantize forward experiment. |
| `fp4_qdq_pertoken` | 11 | NVFP4 per-token quantize/dequantize forward experiment. |
| `fp4_subchannel_scaling_cursor2_cutlass_bwd` | 12 | Cursor2 CUTLASS forward with the MXFP8 CUTLASS dgrad experiment. |
| `int4_marlin_forward` | 13 | Packed INT4 Marlin forward; GroupedLinear requires DISABLE_BACKWARD_QUANT=1. |
| `nvfp4_per_token_w_per_tensor` | 16 | QAT15 activation path with per-expert tensor weight amax; ablation recipe. |

## Backward precision, layout, and existing flags

| Configuration | Effective path |
|---|---|
| `nvfp4_per_token`, no backward override | NVFP4 per-token forward, MXFP8 backward |
| `nvfp4_per_token` + `DISABLE_BACKWARD_QUANT=1` | Layer BF16 bypass; legacy `qat15_disable_bwd` |
| `nvfp4_per_token_bf16_bwd`, no backward override | Dedicated BF16 backward operator |
| QAT17 + `DISABLE_BACKWARD_QUANT=1` | Existing layer bypass overrides the dedicated path; unset the flag to use the dedicated operator |

Descriptions and runtime metadata record `backward` and `backward_source`.
Explicit Python parameters override recipe selection, while existing backward,
layout, and per-layer quantization exclusions continue to apply.

| Setting | Purpose |
|---|---|
| `MINITE_NVFP4_PATH=auto/dense/list` | Per-token TE grouped packaging; ordinary calls default to `auto`, e2e to `dense` |
| `DISABLE_BACKWARD_QUANT=1` | Existing BF16 layer bypass |
| `QAT_KEEP_IN_BF16_LAYER_NUMBERS=0,1` | Disable quantization for selected layers; layer dtype still applies |
| `NVTE_NVFP4_DISABLE_RHT=1` | Disable RHT in recipes that support this toggle |
| `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1` | Disable stochastic rounding in recipes that support this toggle |
| `NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1` | Disable 2D quantization in recipes that support this toggle |
| `INT4_FAKE_GROUP_SIZE` | Fake INT4 group size, default 16; existing aliases remain |
| `INT4_REAL_GROUP_SIZE` / `INT4_REAL_SYMMETRIC` | Existing Marlin settings, defaults 128 / 1 |
| `SHOULD_REPLACE_TE_GROUPLINEAR=1` | Layer replacement in the existing Megatron integration |

QAT15–17 fix RHT and stochastic rounding to disabled; NVTE flags do not enable
them. Layout `legacy` and the old boolean aliases remain accepted. Other strings
retain the historical `auto` behavior, with an inspection note recommending
canonical values. `describe` reports the requested layout; `doctor` checks the
available TE bindings. `auto` selects dense if available, otherwise list.

`recipes` and `describe` require only the standard library, from the repository
root. `doctor` exits 0 for passing prerequisites, 1 for missing prerequisites,
and 2 for invalid arguments. It checks visible GPUs, TE interfaces, and the
selected operator import, without running GEMM or verifying numerical correctness.

## Launch and verification

```bash
bash develop/script/moe_training_e2e.sh \
  --backend mini_nvfp4_per_token --profile off --dry-run
```

See [launcher options](../../script/README.md). The launcher owns backend-specific
selection, clears stale caller selections, and sets matching `MINITE_RECIPE` and
`QAT_PARAMS`. Dry-run prints the configuration and full command, with no log/cache
directories created. Actual runs write `run_meta.json`.

CPU-only compatibility checks, from the repository root:

```bash
python -m unittest discover -s tests -p 'test_recipe_configuration.py' -v
```

Actual configuration and forward/backward equivalence checks in the matched
Blackwell / CUDA / TE environment:

```bash
pytest tests/python/module/test_recipe_api_equivalence.py -v
python tests/python/module/test_group_linear_exact.py --recipe nvfp4_per_token --tests 1 2
```

GPU tests cover independent configuration instances and GroupedLinear selector
parity for QAT4/14/15/16/17, dense Linear parity for QAT4/15, and the separate QAT15 BF16 bypass. The full parameter
construction check also requires reference/experimental operator dependencies.
The older smoke script's `--qparam` is a **test ID, not a QAT_PARAMS ID**: test ID 9
selects QAT15, and test ID 6 selects QAT7. Historical test IDs are preserved; use
`--recipe` for new commands. The two options are mutually exclusive.

Only CPU checks were performed locally for this change. CUDA compilation, GPU
numerical parity, and full training must be validated on the target environment.
Operator implementations and layer interfaces are preserved; operator imports
are deferred until the corresponding recipe is selected.
