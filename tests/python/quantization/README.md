# Quantization Precision Evals

Research scripts that print MAE / SNR / Rel L2 for low-precision formats.
These are not pytest unit tests.

## Files

| File | Entry | What it compares |
|---|---|---|
| `helpers.py` | — | Shared input stats + named-config runner |
| `miniTransformer/ops/exp/quant_utils.py` | — | Lookup tables, block/pertoken quant, metrics |
| `eval_te_nvfp4_vs_mxfp8.py` | `run_nvfp4_vs_mxfp8()` | TE NVFP4 vs MXFP8 under several distributions |
| `eval_exponent_bits.py` | `run_exponent_bits_experiment()` | Fixed M=1, vary E=2/3/4/5 |
| `eval_mantissa_bits.py` | `run_mantissa_bits_experiment()` | Fixed E=2, vary M=1/2/3/4 |
| `eval_pertoken_amax.py` | `run_pertoken_amax_experiment()` | E2Mx + per-token global amax |
| `eval_format_comparison.py` | `run_format_comparison()` | MXFP8 / custom / NVFP4 TE / REF |

## Running

```bash
python eval_exponent_bits.py
python eval_mantissa_bits.py
python eval_pertoken_amax.py
python eval_format_comparison.py

python eval_te_nvfp4_vs_mxfp8.py --shape 1024 4096 --dtype bfloat16
python eval_te_nvfp4_vs_mxfp8.py --dist outlier
python eval_te_nvfp4_vs_mxfp8.py --all-dist
```

## Requirements

- PyTorch with CUDA
- `transformer_engine` for `eval_te_nvfp4_vs_mxfp8.py` and `eval_format_comparison.py`
- `miniTransformer` installed for the REF path in `eval_format_comparison.py`

## Block scale encode / decode

```
decode_scale_q = round(block_amax / fp_max)   # stored as e4m3 or e8m0
encode_scale   = 1 / decode_scale_q           # derived, not stored
x_scaled       = clamp(x * encode_scale, -fp_max, fp_max)
x_q            = quantize(x_scaled)
x_dq           = x_q * decode_scale_q
```

Using `round` (not `ceil`) and deriving `encode_scale` from the quantized
`decode_scale_q` keeps encode/decode invertible. The clamp avoids overflow when
`round(decode_scale) < decode_scale`.

## Metrics

| Metric | Meaning |
|---|---|
| MAE | Mean absolute error |
| Max Err | Max absolute error |
| Rel Err | Mean relative error |
| SNR (dB) | Signal-to-noise; higher is better (~+6 dB per mantissa bit) |
| Rel L2 | Relative L2 error |
