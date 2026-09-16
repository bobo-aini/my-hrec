#!/bin/bash -l
#SBATCH --job-name=hrec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=hrec-%j.log

# Select the local partition at submission: sbatch --partition=<partition>
set -euo pipefail
: "${HREC_ROOT:?Set HREC_ROOT to the extracted HREC directory}"
: "${HREC_PYTHON:?Set HREC_PYTHON to the environment Python executable}"
: "${HREC_CONFIG:?Set HREC_CONFIG to a config path relative to HREC_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
cd "$HREC_ROOT"
"$HREC_PYTHON" -m hrec pipeline --config "$HREC_CONFIG"
