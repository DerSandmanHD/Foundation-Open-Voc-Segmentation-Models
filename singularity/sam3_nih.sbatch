#!/usr/bin/env bash
#SBATCH --job-name=sam3_nih
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --error=sam3_nih.%J.err
#SBATCH --output=sam3_nih.%J.out

set -euo pipefail

SINGULARITY_DIR="${SAM3_SINGULARITY_DIR:-${HOME}/singularity}"
PROJECT_ROOT="${SAM3_PROJECT_ROOT:-${HOME}/Foundation-Open-Voc-Segmentation-Models}"
IMAGE_PATH="${SAM3_SINGULARITY_IMAGE:-${SINGULARITY_DIR}/sam3_master.simg}"
CHECKPOINT_PATH="${SAM3_CHECKPOINT:-${SINGULARITY_DIR}/models/sam3.pt}"

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

if [[ -n "${SAM3_STAGE_DATASET:-}" ]]; then
    SCRATCH_ROOT="/scratch/${SLURM_JOB_ID}"
    mkdir -p "${SCRATCH_ROOT}/images"
    cp -R "${SAM3_STAGE_DATASET}/." "${SCRATCH_ROOT}/images/"
    set -- --image-root "${SCRATCH_ROOT}/images" "$@"
fi

nvidia-smi

singularity exec --nv --bind /scratch:/scratch "${IMAGE_PATH}" \
    python -m sam3_cluster.nih_text_benchmark \
    --checkpoint "${CHECKPOINT_PATH}" \
    --precision bf16 \
    --output-dir "${PROJECT_ROOT}/sam3_outputs/nih_atelectasis" \
    "$@"
