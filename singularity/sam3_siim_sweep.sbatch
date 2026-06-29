#!/usr/bin/env bash
#SBATCH --job-name=sam3_siim_sweep
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --error=sam3_siim_sweep.%J.err
#SBATCH --output=sam3_siim_sweep.%J.out

set -euo pipefail

SINGULARITY_DIR="${SAM3_SINGULARITY_DIR:-${SLURM_SUBMIT_DIR}}"
PROJECT_ROOT="${SAM3_PROJECT_ROOT:-$(cd "${SINGULARITY_DIR}/.." && pwd)}"
IMAGE_PATH="${SAM3_SINGULARITY_IMAGE:-${SINGULARITY_DIR}/sam3_master.simg}"
CHECKPOINT_PATH="${SAM3_CHECKPOINT:-${PROJECT_ROOT}/models/sam3/sam3.pt}"
DATA_ROOT="${SAM3_SIIM_ROOT:-${PROJECT_ROOT}/../data/SIIM_Dataset}"
SWEEP_CSV="${SAM3_SIIM_SWEEP_CSV:-${PROJECT_ROOT}/sam3_cluster/siim_prompt_sweep.csv}"
OUTPUT_ROOT="${SAM3_SIIM_SWEEP_OUTPUT_ROOT:-${PROJECT_ROOT}/sam3_outputs/siim_prompt_sweep}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Singularity-Image fehlt: ${IMAGE_PATH}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "SAM3-Checkpoint fehlt: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/sam3_cluster" ]]; then
    echo "Projektwurzel ist ungültig: ${PROJECT_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "SIIM-Datenordner fehlt: ${DATA_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${SWEEP_CSV}" ]]; then
    echo "SIIM-Sweep-CSV fehlt: ${SWEEP_CSV}" >&2
    exit 1
fi

DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
SWEEP_CSV="$(cd "$(dirname "${SWEEP_CSV}")" && pwd)/$(basename "${SWEEP_CSV}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

echo "Projekt: ${PROJECT_ROOT}"
echo "SIIM-Datenwurzel: ${DATA_ROOT}"
echo "Sweep-CSV: ${SWEEP_CSV}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output-Root: ${OUTPUT_ROOT}"

cd "${PROJECT_ROOT}"
nvidia-smi

singularity exec --nv \
    --bind "${DATA_ROOT}:${DATA_ROOT}:ro" \
    --bind "${OUTPUT_ROOT}:${OUTPUT_ROOT}:rw" \
    "${IMAGE_PATH}" \
    python -m sam3_cluster.run_siim_prompt_sweep \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --sweep-csv "${SWEEP_CSV}" \
    --output-root "${OUTPUT_ROOT}" \
    --precision fp16 \
    --save-examples 10 \
    "$@"
