# miniTransformer

**NVFP4 training through named recipes and Transformer Engine-style linear layers.**

![Version](https://img.shields.io/badge/version-1.0.0-2563eb)

[Quick start](#quick-start) · [Recipes](#choose-a-recipe) · [Megatron integration](#megatron-integration) · [Documentation](#documentation) · [Papers](#papers) · [Development](#development)

miniTransformer is the implementation package for **miniTE**. It provides PyTorch
`Linear`, `LayerNormLinear`, and `GroupedLinear` modules for experimenting with
low-precision training inside an existing model or Megatron workflow. Select a
numerical recipe, pass it to a layer, and inspect the resulting configuration
before running training.

## Papers

- **miniTE: Diagnosing and Bridging the Execution Gap in Native NVFP4
  Reinforcement Learning** — [arXiv:2610.23928](https://arxiv.org/abs/2610.23928).
- **miniTE technical report** (failure diagnosis, TRIAGE, miniTE implementation,
  and measured gains, with interactive figures) —
  [LMSYS blog post](https://www.lmsys.org/blog/2026-10-25-minite).

## Highlights

- **Named precision recipes.** Choose NVFP4 block scaling, per-token NVFP4 with
  MXFP8 backward, or per-token NVFP4 with a dedicated BF16 backward operator.
- **Dense and MoE building blocks.** Use TE-style module interfaces and
  `QLinearParams` for activation, weight, gradient, and GEMM settings. Layer-name
  overrides support selective quantization.
- **Inspectable execution.** List recipes without GPU dependencies, check the
  selected CUDA/TE environment, and preview a complete Megatron launch command.
- **Tools for numerical work.** Collect tensor statistics, export profiler traces,
  and compare intermediate outputs when developing recipes and kernels.

## Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA Blackwell for the native NVFP4 paths, with kernels built for the target architecture |
| CUDA | CUDA Toolkit 12.6 or newer for the package extension |
| PyTorch | A CUDA-enabled build compatible with the training environment |
| Transformer Engine | A matching build exposing the interfaces used by the selected recipe |
| Per-token recipes | `NVFP4Quantizer(per_token=True, per_token_weight_2d=...)` and TE grouped per-token quantization/GEMM bindings |

The per-token interface requirement corresponds to TE PR #3045. Check the actual
bindings with `doctor`; a container tag or TE version label alone does not identify
the complete interface set. See the [environment setup](patch/b200/README.md) for
integration and build details.

## Quick start

### Install from source

In the matched CUDA/PyTorch/TE environment:

```bash
git clone --recurse-submodules https://github.com/fy1214/miniTransformer.git
cd miniTransformer
pip install -e . --no-build-isolation
```

For a checkout cloned without submodules, run
`git submodule update --init --recursive` before installing.

<details>
<summary>Container starting point</summary>

The Slime image provides a starting environment. Complete the matching TE and
Megatron setup before running the NVFP4 examples.

```bash
docker pull slimerl/slime:v0.2.3
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:v0.2.3 /bin/bash
```

</details>

### Inspect a recipe

Run these commands from the repository root. `recipes` and `describe` use only
the Python standard library, so you can explore the configuration before GPU setup.

```bash
python -m miniTransformer recipes
python -m miniTransformer describe nvfp4_per_token
python -m miniTransformer doctor --recipe nvfp4_per_token
```

`doctor` reports missing runtime prerequisites and checks the selected operator's
imports. A successful check establishes dependency availability; forward/backward
correctness is covered by the GPU tests below.

### Run a linear layer

This example performs a forward and backward pass on a CUDA GPU:

```python
import torch
from miniTransformer.module.config import get_qlinear_params
from miniTransformer.module.linear import Linear

recipe = get_qlinear_params("nvfp4_per_token")
layer = Linear(
    1024, 1024,
    bias=False,
    params_dtype=torch.bfloat16,
    device="cuda",
    qlinear_params=recipe,
)

x = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = layer(x)
y.float().square().mean().backward()
```

For an integrated training program, select the same recipe with:

```bash
MINITE_RECIPE=nvfp4_per_token python train.py
```

Here `train.py` is your training entry point. The recipe selects numerical behavior;
your model integration determines which layers use miniTransformer.

## Choose a recipe

| Recipe | Forward | Backward | Use |
|---|---|---|---|
| `nvfp4_per_token` | Per-token NVFP4 | MXFP8 | Starting point for per-token training |
| `nvfp4_per_token_bf16_bwd` | Per-token NVFP4 | Dedicated BF16 operator | Use BF16 backward GEMMs and saved operands |
| `nvfp4_cursor2_cutlass_v2` | Per-token activations, tensor weight amax; CUTLASS V2 | MXFP8 | Custom grouped MoE path |
| `nvfp4_block` | NVFP4 block scaling | NVFP4 | Block-scaling experiments |
| `unquantized` | Layer dtype | Layer dtype | High-precision comparison |

Each `get_qlinear_params` call constructs independent configuration and operator
instances. Explicit `qlinear_params` take priority over the `MINITE_RECIPE` selector;
existing precision and layout tuning flags still apply.

Use `python -m miniTransformer recipes --all` for reference and experimental
recipes. The [recipe guide](develop/docs/en/recipes.md) explains scale granularity,
backward overrides, dense/list packaging, and per-layer configuration. In particular,
the dedicated BF16 recipe and the layer-level BF16 bypass have different execution paths.

## Megatron integration

miniTransformer integrates at the linear-layer boundary:

| Megatron wrapper | miniTransformer module |
|---|---|
| `TELinear` | `miniTransformer.module.linear.Linear` |
| `TELayerNormColumnParallelLinear` | `miniTransformer.module.layernorm_linear.LayerNormLinear` |
| `TEGroupedLinear` | `miniTransformer.module.group_linear.GroupedLinear` |

Use the matching Megatron wrappers or the [integration setup](patch/b200/README.md).
The MoE launcher sets the layer-replacement switch and recipe together for its
miniTransformer backends. Recipe selection alone does not modify model classes.

Preview the configuration and complete command first:

```bash
bash develop/script/moe_training_e2e.sh \
  --backend mini_nvfp4_per_token --profile off --dry-run
```

Then run in the prepared training environment:

```bash
MEGATRON_ROOT=/root/Megatron-LM TRAIN_ITERS=30 \
  bash develop/script/moe_training_e2e.sh \
  --backend mini_nvfp4_per_token --profile off
```

The launcher uses mock data when `DATA_PATH` is empty. See the [launcher guide](develop/script/README.md)
for real data, parallel settings, BF16 comparisons, and profiling. Normal runs
record the recipe, forward/backward settings, layout request, and configuration
source in `run_meta.json`. `--dry-run` creates no training, log, or cache directories.

## Documentation

| Topic | English | 中文 |
|---|---|---|
| Complete recipe guide | [Recipes](develop/docs/en/recipes.md) | [配置手册](develop/docs/zh/recipes.md) |
| Architecture and extension workflow | [Development](develop/docs/en/develop.md) | [开发指南](develop/docs/zh/develop.md) |
| Train–inference deterministic alignment | [Deterministic mode](develop/docs/en/nvfp4-deterministic-mode.md) | [确定性对齐](develop/docs/zh/nvfp4-deterministic-mode.md) |

Additional entry points: [environment setup](patch/b200/README.md),
[MoE launcher](develop/script/README.md), [NVFP4 kernel tests](tests/python/nvfp4/README.md),
[distributed tests](tests/python/distributed/README.md), and
[log-probability comparison tools](tools/logpdiff-tools/README.md).

## Development

Configuration and launcher checks require no GPU packages:

```bash
python -m unittest discover -s tests -p 'test_recipe_configuration.py' -v
```

Run numerical checks in the matched Blackwell environment:

```bash
pytest tests/python/module/test_recipe_api_equivalence.py -v
python tests/python/module/test_group_linear_exact.py \
  --recipe nvfp4_per_token --tests 1 2
```

To add a recipe, implement the quantization/GEMM interfaces, register its name and
metadata, and connect it to the parameter factory. Follow the
[extension workflow](develop/docs/en/develop.md#52-register-recipe-and-index)
for operator, module, and training validation.

For a bug report or pull request, include the recipe description, `doctor` output,
GPU and CUDA/TE environment, and a minimal reproducer. Remove private model/data
paths from shared logs. Report configuration checks, numerical checks, and
full-training results separately.

<details>
<summary>Common setup questions</summary>

**Can I inspect recipes on a machine without CUDA?** Yes. From the repository root,
`recipes` and `describe` work without Torch or TE. Running the layers needs the
CUDA training environment.

**Why does a per-token recipe fail on an installed TE package?** The selected path
needs specific quantizer arguments and grouped bindings. Run `doctor` to identify
the missing interface, then use the matching TE build described in the setup guide.

**Why is a MoE integration rejected?** Check the grouped layout and output-dimension
requirements. `GroupedLinear` currently does not support sequence parallelism.
Checkpoint parameter loading and restoration of TE's internal quantization state
are separate compatibility concerns.

</details>
