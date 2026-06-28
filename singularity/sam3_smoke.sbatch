#!/usr/bin/env bash
#SBATCH --job-name=sam3_smoke
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=00:15:00
#SBATCH --error=sam3_smoke.%J.err
#SBATCH --output=sam3_smoke.%J.out

set -euo pipefail

SINGULARITY_DIR="${SAM3_SINGULARITY_DIR:-${SLURM_SUBMIT_DIR}}"
PROJECT_ROOT="${SAM3_PROJECT_ROOT:-$(cd "${SINGULARITY_DIR}/.." && pwd)}"
IMAGE_PATH="${SAM3_SINGULARITY_IMAGE:-${SINGULARITY_DIR}/sam3_master.simg}"
CHECKPOINT_PATH="${SAM3_CHECKPOINT:-${PROJECT_ROOT}/models/sam3/sam3.pt}"
DEFAULT_IMAGE="${PROJECT_ROOT}/Foundation & Open-Vocabulary Segmentation Models/sam3/assets/images/truck.jpg"

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

cd "${PROJECT_ROOT}"
nvidia-smi

singularity exec --nv "${IMAGE_PATH}" \
    python -m sam3_cluster.smoke_test \
    --image "${DEFAULT_IMAGE}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --precision fp16 \
    --output-dir "${PROJECT_ROOT}/sam3_outputs/sam3_smoke_test" \
    "$@"
