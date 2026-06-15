# TransformerEngine patch — disable FP8 fallback

When training with `--fp4`, TE applies quantization to every `Linear` /
`LayerNormLinear` instance.  For layers that are NOT replaced by
miniTransformer's grouped-linear (i.e. everything outside the MoE experts),
we want them to stay in bf16.  Easiest way: assign `fp8 = False` at the top
of `_Linear.forward` and `_LayerNormLinear.forward`, right after the
`non_tensor_args` tuple is unpacked.

## Why a Python script, not a .patch file

The TE version pinned inside `slimerl/slime:v0.2.3` already refactored
`forward()` to take a single `non_tensor_args: Tuple` and unpack it inline.
Older TE checkouts (and what's currently on github main) still have the
arguments spread out as named kwargs.  A strict unified diff would only apply
to one of the two layouts.  The script below is idempotent and matches on the
`) = non_tensor_args` marker, so it works on any version that uses the
container's refactored signature and is a no-op once applied.

## What it does

Inserts these two lines after `) = non_tensor_args`:

```python
    # miniTE-b200: disable FP8 fallback for unreplaced Linear layers
    fp8 = False
```

A `.miniTE.bak` backup is written next to each patched file the first time.

## Usage

```bash
python apply_disable_fp8.py
# or, if TE is installed somewhere unusual:
python apply_disable_fp8.py --te-dir /usr/local/lib/python3.12/dist-packages/transformer_engine
```

Rerunning is safe — files are checked for the sentinel comment before any edit.
