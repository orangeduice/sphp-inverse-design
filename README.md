# sphp-inverse-design

Data-driven inverse design of infrared surface-phonon-polariton (SPhP) 4H-SiC micropillar metasurfaces, using a
conditional variational autoencoder (cVAE). Part of the MINAS project (WP2, Milestone 4 / Deliverable 5).

**Status:** being set up. The code is being consolidated from earlier versions, which are archived in
[orangeduice/MINAS-inverse-design](https://github.com/orangeduice/MINAS-inverse-design).

## Layout (planned)

| Path | Contents |
|---|---|
| `src/` | data loading, models (geometry cVAE; optional spectra cVAE; forward model), train / infer / metrics |
| `configs/` | one YAML per experiment |
| `comsol/` | COMSOL models and SLURM job scripts (run on the Iridis HPC only); see `comsol/README.md` |
| `figures/` | one script per thesis/paper figure |
| `tests/` | parity and smoke tests |

## Setup

1. Copy `config.example.json` to `config.json` and fill in your HPC details. `config.json` is git-ignored.
2. The environment and the rest of the setup are still to come.

## Related work

- Tandem-network baseline: [orangeduice/MINAS-ivd-iter](https://github.com/orangeduice/MINAS-ivd-iter)
