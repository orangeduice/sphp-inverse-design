# sphp-inverse-design

Data-driven inverse design of infrared surface-phonon-polariton (SPhP) 4H-SiC micropillar metasurfaces, using a
conditional variational autoencoder (cVAE). Part of the MINAS project (WP2, Milestone 4 / Deliverable 5).

**Status:** being set up. The code is being consolidated from earlier versions, which are archived in
[orangeduice/MINAS-inverse-design](https://github.com/orangeduice/MINAS-inverse-design).

## Layout (planned)

| Path | Contents |
|---|---|
| `src/sphp_inverse/` | data loading, models (geometry cVAE; optional spectra cVAE; forward model), train / infer / metrics |
| `configs/` | one YAML per experiment |
| `comsol/` | COMSOL models and SLURM job scripts (run on the Iridis HPC only); see `comsol/README.md` |
| `figures/` | one script per thesis/paper figure |
| `tests/` | parity and smoke tests |
| `parity/` | export of the Keras 2.15 reference model |

## Setup

1. Install [uv](https://docs.astral.sh/uv/). Then run `uv sync`. This installs Python 3.12 and PyTorch with CUDA 12.8,
   which supports Blackwell GPUs such as the RTX 5070 Ti.
2. Copy `config.example.json` to `config.json` and fill in your HPC details. `config.json` is git-ignored.
3. The dataset is not in git. Tests look for `../data/2026_04_15_full_pisa_data.h5` by default; override this with
   `SPHP_DATA=<path>`.
4. Run the tests: `uv run --group dev pytest`.

## Status of the PyTorch port

The geometry cVAE in `src/sphp_inverse/models/cvae.py` reproduces the original Keras 2.15 model exactly when given
its weights, and trains to the same accuracy with the same recipe. See `parity/README.md`.

## Workflow: train → suggest designs → re-simulate → score

```bash
uv run python -m sphp_inverse.train --config configs/geometry_cvae_reference.yaml --data <dataset.h5> --out <run dir> --seed 0
```

```bash
uv run python -m sphp_inverse.inference --model <run dir> --data <dataset.h5> --test-idx 138 169 --n 5 --out <inference dir> --category 5
```

```bash
uv run python -m sphp_inverse.hpc submit <inference dir> --mph 2026_03_02_micropillar_resim_v64.mph --dry-run
```

```bash
uv run python -m sphp_inverse.hpc fetch <inference dir>
```

```bash
uv run python -m sphp_inverse.resim <inference dir>
```

- `inference` writes one `run_<ID>/` folder per target, containing `<ID>_params.csv` (the designs COMSOL will
  simulate), the target spectrum and a metadata file. The metadata records the z samples and validity flags:
  inside the training bounds, aspect ratio, and pillar gap.
- `hpc submit` uploads the job scripts and every parameter file, then submits one SLURM job per run. This takes two
  SSH connections in total. Remove `--dry-run` to really submit. `hpc status` shows your queue.
- `hpc fetch` downloads `<ID>_dataout.csv` into each run folder that doesn't have it yet. It searches the
  `results_dirs` listed in `config.json`.
- Transfers use `rsync` when it is installed. Without it (e.g. plain Windows) they fall back to `scp` and an
  `ssh … tar` stream.
- The COMSOL models have their file locations hard-coded: the resim model reads `~/micropillar_resim/<ID>_params.csv`,
  so `remote_dir` in `config.json` must point there.

## Related work

- Tandem-network baseline: [orangeduice/MINAS-ivd-iter](https://github.com/orangeduice/MINAS-ivd-iter)
