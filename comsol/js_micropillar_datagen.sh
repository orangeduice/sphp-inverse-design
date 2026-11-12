#!/bin/bash
# Generate training data (micropillar reflection spectra) with COMSOL 6.4 on Iridis.
#
# Usage (run from the synced comsol/ folder, or let the job submitter do it):
#   sbatch [--mail-user=<email> --mail-type=ALL] js_micropillar_datagen.sh <RUN_ID> [MPH_FILE] [OUT_DIR]
#
#   RUN_ID    data-generation batch id; COMSOL's set_params_call samples/reads that batch's parameters
#   MPH_FILE  datagen model (default: 2026_03_02_micropillar_datagen_v64.mph, in this folder)
#   OUT_DIR   where the temporary solved .mph is written (default: /scratch/$USER/COMSOL_pisa_gen)
#
# Original (hard-coded) version kept outside this repo: js_micropillar_data_gen_dis.sh
# (its commented-out node-export / compression steps are not carried over).

#SBATCH --job-name=COMSOL_datagen
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=32
#SBATCH --time=32:00:00
#SBATCH -L comsolbatch@comsollm:1

set -eu

RUN_ID=${1:?Usage: sbatch js_micropillar_datagen.sh <RUN_ID> [MPH_FILE] [OUT_DIR]}
MPH_FILE=${2:-2026_03_02_micropillar_datagen_v64.mph}
OUT_DIR=${3:-/scratch/$USER/COMSOL_pisa_gen}

cd "$SLURM_SUBMIT_DIR"
case "$MPH_FILE" in
  /*) INPUT_PATH="$MPH_FILE" ;;
  *)  INPUT_PATH="$SLURM_SUBMIT_DIR/$MPH_FILE" ;;
esac
[ -f "$INPUT_PATH" ] || { echo "Model not found: $INPUT_PATH"; exit 1; }
mkdir -p "$OUT_DIR"
OUTPUT_PATH="$OUT_DIR/${RUN_ID}_micropillar_out.mph"

echo "run=$RUN_ID model=$INPUT_PATH out=$OUTPUT_PATH"
module load comsol/6.4

# 1. Set parameters for this batch
comsol batch -nosplash -nodesktop -autosave off \
    -inputfile "$INPUT_PATH" -outputfile "$OUTPUT_PATH" \
    -pname ID -plist "$RUN_ID" -methodcall set_params_call

# 2. Solve (distributed)
comsol batch -mpibootstrap slurm -nosplash -nodesktop -autosave off \
    -nn "$SLURM_NTASKS" -np "$SLURM_CPUS_PER_TASK" \
    -inputfile "$OUTPUT_PATH" -usebatchlic

# 3. Export data
comsol batch -nosplash -nodesktop -autosave off \
    -inputfile "$OUTPUT_PATH" -pname ID -plist "$RUN_ID" -methodcall data_out_call

# 4. Remove the large solved .mph (as in the original script)
rm -v "$OUTPUT_PATH"
