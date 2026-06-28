#!/usr/bin/env bash
#SBATCH --job-name=sam31_smoke
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=00:30:00
#SBATCH --error=sam31_smoke.%J.err
#SBATCH --output=sam31_smoke.%J.out

set -euo pipefail

SINGULARITY_DIR="${SAM3_SINGULARITY_DIR:-${SLURM_SUBMIT_DIR}}"
PROJECT_ROOT="${SAM3_PROJECT_ROOT:-$(cd "${SINGULARITY_DIR}/.." && pwd)}"
IMAGE_PATH="${SAM3_SINGULARITY_IMAGE:-${SINGULARITY_DIR}/sam3_master.simg}"
CHECKPOINT_PATH="${SAM31_CHECKPOINT:-${PROJECT_ROOT}/models/sam3.1/sam3.1_multiplex.pt}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Singularity-Image fehlt: ${IMAGE_PATH}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "SAM3.1-Checkpoint fehlt: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/sam3_cluster" ]]; then
    echo "Projektwurzel ist ungültig: ${PROJECT_ROOT}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
nvidia-smi

singularity exec --nv "${IMAGE_PATH}" \
    python -m sam3_cluster.sam31_smoke_test \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output-dir "${PROJECT_ROOT}/sam3_outputs/sam31_smoke_test" \
    "$@"
