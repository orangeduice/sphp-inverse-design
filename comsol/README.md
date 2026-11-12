# COMSOL models and SLURM job scripts

COMSOL is **not installed on the local workstation**. These run only on Iridis 6 (`module load comsol/6.4`).
Locally we only prepare design/parameter CSVs and parse the exported `*_dataout.csv` results.

| File | Purpose |
|---|---|
| `2026_03_02_micropillar_datagen_v64.mph` | Dataset generation model (md5 `afb51606…`) |
| `2026_03_02_micropillar_resim_v64.mph` | Re-simulation, **full range** 25–29 THz (md5 `a4d3e6d5…`) |
| `2026_06_29_micropillar_resim_rr_v64.mph` | Re-simulation, **reduced range**. ⚠️ Still on Iridis only: copy it here |
| `js_micropillar_datagen.sh` | SLURM job: set params → solve (6 × 32 CPUs, 32 h) → export → delete solved `.mph` |
| `js_micropillar_resim.sh` | SLURM job: set params → solve (4 h) → export; keeps the solved `.mph` |

`.mph` files are git-ignored for now. Whether to publish the COMSOL models is still to be decided.

## How jobs are sent

The repo is the single source of truth. The job submitter first `rsync`s this `comsol/` folder (the scripts plus
the `.mph` files) to the remote directory in `config.json`, then runs `sbatch` there:

```bash
sbatch --mail-user=<email> --mail-type=ALL js_micropillar_resim.sh <RUN_ID> 2026_06_29_micropillar_resim_rr_v64.mph
```

Nothing user-specific is hard-coded. Output goes to `/scratch/$USER/...` unless an `OUT_DIR` is given, and the
e-mail address comes from `config.json` at submit time.

The original hard-coded scripts are kept for provenance in the parent working folder (outside this repo).
