#!/usr/bin/env bash
#SBATCH --job-name=sam3_siim
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --error=sam3_siim.%J.err
#SBATCH --output=sam3_siim.%J.out

set -euo pipefail

SINGULARITY_DIR="${SAM3_SINGULARITY_DIR:-${SLURM_SUBMIT_DIR}}"
PROJECT_ROOT="${SAM3_PROJECT_ROOT:-$(cd "${SINGULARITY_DIR}/.." && pwd)}"
IMAGE_PATH="${SAM3_SINGULARITY_IMAGE:-${SINGULARITY_DIR}/sam3_master.simg}"
CHECKPOINT_PATH="${SAM3_CHECKPOINT:-${PROJECT_ROOT}/models/sam3/sam3.pt}"
DATA_ROOT="${SAM3_SIIM_ROOT:-${PROJECT_ROOT}/../data/SIIM_Dataset}"
OUTPUT_DIR="${SAM3_SIIM_OUTPUT_DIR:-${PROJECT_ROOT}/sam3_outputs/siim_pneumothorax}"

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

DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
echo "Projekt: ${PROJECT_ROOT}"
echo "SIIM-Datenwurzel: ${DATA_ROOT}"
echo "Checkpoint: ${CHECKPOINT_PATH}"

cd "${PROJECT_ROOT}"
nvidia-smi

singularity exec --nv --bind "${DATA_ROOT}:${DATA_ROOT}:ro" "${IMAGE_PATH}" \
    python -m sam3_cluster.siim_text_benchmark \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --text-prompt "pneumothorax" \
    --precision fp16 \
    --save-examples 10 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
