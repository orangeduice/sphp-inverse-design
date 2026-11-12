#!/bin/bash
# Re-simulate suggested micropillar designs with COMSOL 6.4 on Iridis.
#
# Usage (run from the synced comsol/ folder, or let the job submitter do it):
#   sbatch [--mail-user=<email> --mail-type=ALL] js_micropillar_resim.sh <RUN_ID> <MPH_FILE> [OUT_DIR]
#
#   RUN_ID    inference run id; COMSOL's set_params_call reads that run's design CSV
#   MPH_FILE  resim model, relative to this folder or absolute, e.g.
#               2026_03_02_micropillar_resim_v64.mph      (full range, 25-29 THz)
#               2026_06_29_micropillar_resim_rr_v64.mph   (reduced range)
#   OUT_DIR   where the solved .mph is written (default: /scratch/$USER/COMSOL_pisa_resim)
#
# Original (hard-coded) version kept outside this repo: js_micropillar_resim_dis.sh

#SBATCH --job-name=COMSOL_resim
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH -L comsolbatch@comsollm:1

set -eu

RUN_ID=${1:?Usage: sbatch js_micropillar_resim.sh <RUN_ID> <MPH_FILE> [OUT_DIR]}
MPH_FILE=${2:?Give the resim .mph file (full-range or reduced-range rr)}
OUT_DIR=${3:-/scratch/$USER/COMSOL_pisa_resim}

cd "$SLURM_SUBMIT_DIR"
case "$MPH_FILE" in
  /*) INPUT_PATH="$MPH_FILE" ;;
  *)  INPUT_PATH="$SLURM_SUBMIT_DIR/$MPH_FILE" ;;
esac
[ -f "$INPUT_PATH" ] || { echo "Model not found: $INPUT_PATH"; exit 1; }
mkdir -p "$OUT_DIR"
OUTPUT_PATH="$OUT_DIR/${RUN_ID}_micropillar_resim_out.mph"

echo "run=$RUN_ID model=$INPUT_PATH out=$OUTPUT_PATH"
module load comsol/6.4

# 1. Set parameters for this run
comsol batch -nosplash -nodesktop -autosave off \
    -inputfile "$INPUT_PATH" -outputfile "$OUTPUT_PATH" \
    -pname ID -plist "$RUN_ID" -methodcall set_params_call

# 2. Solve (distributed)
comsol batch -mpibootstrap slurm -nosplash -nodesktop -autosave off \
    -nn "$SLURM_NTASKS" -np "$SLURM_CPUS_PER_TASK" \
    -inputfile "$OUTPUT_PATH" -usebatchlic

# 3. Export reflection spectra (<RUN_ID>_dataout.csv)
comsol batch -nosplash -nodesktop -autosave off \
    -inputfile "$OUTPUT_PATH" -pname ID -plist "$RUN_ID" -methodcall data_out_call

# The solved .mph is kept (as in the original script) for inspection.
