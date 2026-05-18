# Quantization Precision Tests

Research scripts for comparing low-precision floating-point quantization formats.
These are not unit tests — they print per-format precision metrics (MAE, SNR, Rel L2)
to inform quantization format selection for miniTransformer.

## Files

| File | Entry point | Experiment |
|---|---|---|
| `_quant_utils.py` | — | Shared utilities (lookup tables, scale casts, block quant, metrics) |
| `test_quant_precision.py` | `run_nvfp4_vs_mxfp8()` | NVFP4 vs MXFP8 via TransformerEngine — baseline comparison |
| `test_quantize_E_effect.py` | `run_exponent_bits_experiment()` | Fixed M=1, vary E=2/3/4/5 — effect of exponent bits |
| `test_quantize_M_effect.py` | `run_mantissa_bits_experiment()` | Fixed E=2, vary M=1/2/3/4 — effect of mantissa bits |
| `test_quantize_pertoken_amax.py` | `run_pertoken_amax_experiment()` | E2Mx with per-token global amax — two-level quantization |
| `test_multi_quant_method.py` | `run_format_comparison()` | Full comparison: MXFP8 / custom formats / NVFP4 / TE / REF |

## Running

```bash
python test_quantize_E_effect.py
python test_quantize_M_effect.py
python test_quantize_pertoken_amax.py
python test_multi_quant_method.py

# test_quant_precision.py supports CLI options
python test_quant_precision.py --shape 1024 4096 --dtype bfloat16
python test_quant_precision.py --dist outlier
python test_quant_precision.py --all-dist
```

## Requirements

- PyTorch with CUDA
- `transformer_engine` (for `test_quant_precision.py` and `test_multi_quant_method.py`)
- `miniTransformer` package installed (for the REF scheme in `test_multi_quant_method.py`)

## Design: block scale encode/decode

All custom implementations follow the same encode/decode convention
(mirrors `nvfp4_quantize_op_ref._quantize_blockwise_reference`):

```
decode_scale_q = round-to-nearest(block_amax / fp_max)   # stored as e4m3 or e8m0
encode_scale   = 1 / decode_scale_q                       # derived, not stored
x_scaled       = clamp(x * encode_scale, -fp_max, fp_max)
x_q            = quantize(x_scaled)
x_dq           = x_q * decode_scale_q
```

Using `round` (not `ceil`) and deriving `encode_scale` from the quantized
`decode_scale_q` ensures strict encode/decode invertibility.
The clamp guards against overflow when `round(decode_scale) < decode_scale`.

## Metrics

| Metric | Description |
|---|---|
| MAE | Mean absolute error |
| Max Err | Maximum absolute error (outlier sensitivity) |
| Rel Err | Mean relative error |
| SNR (dB) | Signal-to-noise ratio — higher is better; each mantissa bit ≈ +6 dB |
| Rel L2 | Relative L2 norm error |

---

## Issue 2: Effect of E2M1 base + E4M3 scale

### Experiment 2.1 — Effect of exponent bits (E) on precision

Fixed mantissa M=1, varied exponent E=2/3/4/5, crossed with two block scale types.
8 schemes total.

**Synthetic data results (random mixed distribution):**

```
Format + Scale                 MAE       SNR(dB)
------------------------------------------------
E5M1  | E4M3 scale          1.485030      0.00
E4M1  | E4M3 scale          0.132963     20.48
E3M1  | E4M3 scale          0.127780     20.63
E2M1  | E4M3 scale          0.154619     19.62
E5M1  | E8M0 scale          0.128281     20.18
E4M1  | E8M0 scale          0.128281     20.18
E3M1  | E8M0 scale          0.128515     20.18
E2M1  | E8M0 scale          0.176643     17.97
```

SNR summary (dB):

| Base \ Scale | E4M3  | E8M0  |
|---|---|---|
| E2M1 | 19.62 | 17.97 |
| E3M1 | 20.63 | 20.18 |
| E4M1 | 20.48 | 20.18 |
| E5M1 | 0.00 (max exceeds E4M3 range) | 20.18 |

**Real inference data (36 layers of QKV linear input, prefill + 1 decode token, averaged):**

SNR (dB):

| Base \ Scale | E4M3  | E8M0  |
|---|---|---|
| E2M1 | 19.88 | 17.11 |
| E3M1 | 21.50 | 19.7  |
| E4M1 | 14.39 | 19.7  |
| E5M1 | 0.00 (overflow) | 19.7 |

Rel_L2:

| Base \ Scale | E4M3    | E8M0    |
|---|---|---|
| E2M1 | 0.10140 | 0.13950 |
| E3M1 | 0.08478 | ~0.103  |
| E4M1 | 0.28937 | ~0.103  |
| E5M1 | 0.99698 (overflow) | ~0.103 |

**RL training: logp diff at step 0 (E4M3 scale, gbs=4):**

| Format | logp diff |
|---|---|
| E2M1 | 0.0580 |
| E3M1 | 0.1152 |
| E4M1 | 3.4122 |
| E5M1 | nan |

> Wider exponent (E) does not improve training-inference consistency and can destabilize training.

---

### Experiment 2.2 — Effect of mantissa bits (M) on precision

Fixed exponent E=2, varied mantissa M=1/2/3/4, crossed with two block scale types.
8 schemes total.

**Synthetic data results (random mixed distribution):**

```
Format + Scale                 MAE       SNR(dB)
------------------------------------------------
E2M4  | E4M3 scale          0.017888     36.91
E2M3  | E4M3 scale          0.034627     31.58
E2M2  | E4M3 scale          0.070138     26.04
E2M1  | E4M3 scale          0.142116     20.17
E2M4  | E8M0 scale          0.018970     37.03
E2M3  | E8M0 scale          0.038432     30.93
E2M2  | E8M0 scale          0.079777     24.93
E2M1  | E8M0 scale          0.176643     17.97
```

SNR summary (dB):

| Base \ Scale | E4M3  | E8M0  |
|---|---|---|
| E2M1 | 20.17 | 17.97 |
| E2M2 | 26.04 | 24.93 |
| E2M3 | 31.58 | 30.93 |
| E2M4 | 36.91 | 37.03 |

**Real inference data (36 layers, averaged):**

SNR (dB):

| Base \ Scale | E4M3  | E8M0  |
|---|---|---|
| E2M1 | 19.88 | 17.11 |
| E2M2 | 26.23 | 23.98 |
| E2M3 | 31.19 | 30.31 |
| E2M4 | 34.96 | 36.41 |

Rel_L2:

| Base \ Scale | E4M3    | E8M0    |
|---|---|---|
| E2M1 | 0.10140 | 0.13950 |
| E2M2 | 0.04885 | 0.06322 |
| E2M3 | 0.02767 | 0.03051 |
| E2M4 | 0.01807 | 0.01513 |

**RL training: logp diff at step 0 (E4M3 scale, gbs=4):**

| Format | logp diff |
|---|---|
| E2M1 | 0.0580 |
| E2M2 | 0.0424 |
| E2M3 | 0.0356 |
| E2M4 | 0.0318 |

> More mantissa bits consistently improve training-inference consistency.

---

### Combined results (Exp 2.1 + 2.2)

Ranked by SNR mean (real inference data, descending):

| Format | SNR mean | SNR worst | MAE mean | Rel_L2 mean |
|---|---|---|---|---|
| E2M4 + E8M0 | 36.41 | 35.87 | 0.007954 | 0.01513 |
| E2M4 + E4M3 | 34.96 | 29.58 | 0.007589 | 0.01807 |
| E2M3 + E4M3 | 31.19 | 28.12 | 0.014404 | 0.02767 |
| E2M3 + E8M0 | 30.31 | 29.73 | 0.016283 | 0.03051 |
| E2M2 + E4M3 | 26.23 | 25.03 | 0.027865 | 0.04885 |
| E2M2 + E8M0 | 23.98 | 23.64 | 0.034200 | 0.06322 |
| E3M1 + E4M3 | 21.50 | 17.81 | 0.045411 | 0.08478 |
| E2M1 + E4M3 | 19.88 | 19.53 | 0.058788 | 0.10140 |
| E5M1/E4M1/E3M1 + E8M0 | ~19.7 | ~19.3 | ~0.051 | ~0.103 |
| E2M1 + E8M0 | 17.11 | 16.66 | 0.077184 | 0.13950 |
| E4M1 + E4M3 | 14.39 | 0.71 | 0.067209 | 0.28937 |
| E5M1 + E4M3 | 0.03 | 0.00 | 0.577118 | 0.99698 |

---

### Key findings

**1. E bits are irrelevant under E8M0 scale**

E5M1 / E4M1 / E3M1 + E8M0 all reach ~19.7 dB SNR — effectively identical.
The bottleneck is M=1 (only 2 mantissa steps), not the exponent range.

**2. Large-E formats are unstable with E4M3 scale**

E4M1+E4M3 SNR worst case drops to 0.71 dB on some activation files; E5M1+E4M3
collapses entirely (fp_max exceeds E4M3 representable range).
More exponent bits is not just unhelpful — it actively breaks E4M3 block scale.

**3. More mantissa bits outperform more exponent bits**

- E2M4 (6-bit data) achieves ~35–36 dB, vs E3M1 (5-bit) at ~21 dB: +15 dB gain
- E2M2 (5-bit) at ~24–26 dB still beats E3M1/E2M1 despite equal bit count
- Each additional M bit improves SNR by ~6 dB, consistent with information theory

**4. E8M0 scale is more stable at high M; E4M3 scale has slightly higher mean but more variance**

- E2M4+E8M0 (SNR 36.41, worst 35.87) beats E2M4+E4M3 (34.96, worst 29.58)
- E2M3+E4M3 (31.19) vs E2M3+E8M0 (30.31): E4M3 wins on mean but has worse floor
- For production use, E8M0 scale is the safer choice at high mantissa formats
