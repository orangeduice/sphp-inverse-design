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

## 3. Training parity (Phase 3a step 4)

Both frameworks were trained from scratch with the notebook's settings (`configs/geometry_cvae_reference.yaml`):
AdamW (lr 1e-4, weight decay 0.004, eps 1e-7), β = 0.001, persistent ReduceLROnPlateau (×0.66, patience 5) and
EarlyStopping (patience 15), batch 64 for 25 epochs then batch 128 for 12 epochs.

```bash
# Keras 2.15 (CPU, about 5 min per run)
uv run --no-project --python 3.11 --with tensorflow==2.15.0 --with h5py --with scikit-learn python parity/train_keras_reference.py --data <h5> --out ../runs/keras_train_seed0 --seed 0
```

```bash
# PyTorch
uv run python -m sphp_inverse.train --config configs/geometry_cvae_reference.yaml --data <h5> --out ../runs/torch_parity_seed0 --seed 0
```

```bash
# comparison table + plot
uv run python parity/compare_training.py --runs ../runs --notebook ../runs/keras_reference/notebook_history.csv
```

### Result (2026-11-12; 3 Keras seeds, 5 PyTorch seeds, mean ± std)

| Metric | Keras 2.15 | PyTorch | Original notebook (1 run) |
|---|---|---|---|
| Test reconstruction, z = μ | 0.0062 ± 0.0003 | 0.0064 ± 0.0011 | – |
| Final validation reconstruction | 0.0070 ± 0.0002 | 0.0075 ± 0.0010 | 0.0073 |
| Best validation loss | 0.0104 ± 0.0001 | 0.0104 ± 0.0009 | 0.0107 |
| Final train reconstruction | 0.0050 ± 0.0011 | 0.0049 ± 0.0012 | 0.0062 |
| Final KL term (× β) | 0.0035 | 0.0035 | 0.0034 |
| Test KL (sum over 8 dims, nats) | 28.6 ± 0.2 | 27.9 ± 1.1 | – |
| Active latent units (Var μ > 0.01) | 8 / 8 | 8 / 8 | – |
| Validation loss at epoch 3 (BatchNorm warm-up) | 0.634 | 0.626 | 0.628 |
| Test reconstruction, z ~ N(0, I) | 0.53 ± 0.03 | 0.51 ± 0.02 | – |

The means agree within the seed-to-seed spread on every metric: there is no sign of the posterior collapse or
overfitting that affected the earlier PyTorch attempt. PyTorch shows a somewhat larger spread between seeds
(3 vs 5 runs is too few to say whether that is real).

Still to do for the final decision (PyTorch vs Keras): re-simulate designs from the PyTorch model in COMSOL on the same targets as the
Keras model.
