# Keras → PyTorch parity

The PyTorch geometry cVAE (`src/sphp_inverse/models/cvae.py`) is a translation of the working Keras 2.15 model.
This folder holds the tool that freezes the Keras model as a reference. `tests/test_keras_parity.py` then checks
the PyTorch model against it.

## 1. Export the Keras reference (TF 2.15, CPU, its own throwaway environment)

TF 2.15 can't use the RTX 5070 Ti and needs Python ≤ 3.11; this is why the earlier Keras 3 upgrade happened.
The export only runs forward passes, so CPU is fine:

```bash
uv run --no-project --python 3.11 --with tensorflow==2.15.0 --with h5py --with scikit-learn python parity/export_keras_reference.py --models <folder with encoder_/decoder_cVAE_geomerty_*_latent8_blocks1.h5> --data <2026_04_15_full_pisa_data.h5> --out ../runs/keras_reference
```

Outputs (outside git):
- `weights.npz` and `layers.json`: every layer's weights and config;
- `*_summary.txt`: the Keras model summaries;
- `batch.npz`: 64 fixed test samples with the Keras outputs, in inference and BatchNorm-training mode, plus the losses;
- `split.json`: the train/val/test indices exactly as Keras computes them;
- `env.json`: versions.

## 2. Run the parity tests (main environment)

```bash
uv run --group dev pytest
```

Override the locations with `SPHP_DATA=<h5>` and `SPHP_KERAS_REF=<export dir>`.

## Result (2026-11-12, reference = `geomerty_cVAE_best/model_latent8_blocks1`)

| Check | Result |
|---|---|
| Parameter counts (encoder 1,953,968; decoder 3,217,825) | identical |
| Encoder μ, log σ² / decoder output, inference mode, float64 | max abs diff 5e-6 (values up to ~8) |
| Same with BatchNorm in training mode (batch statistics) | ✔ |
| Loss terms (reconstruction, β·KL) | ✔ (rtol 1e-4) |
| Train/val/test split vs Keras' own `validation_split` | identical (6517 / 2793 / 490) |
| float32 on the RTX 5070 Ti | 7e-6 with TF32 off; **6e-3 with TF32 on** (PyTorch's cuDNN default) |
| Negative control: channels-first flatten | fails by 7.3: the test does catch layout bugs |

Note: with 161 frequency points every padded length is odd, so Keras 'same' padding happens to be symmetric.
It isn't for the reduced-range (75-point) spectra. `SameConv1d` handles both; see `tests/test_layers.py`.
