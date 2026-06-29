#!/usr/bin/env bash
#SBATCH --job-name=sam3_nih
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --error=sam3_nih.%J.err
#SBATCH --output=sam3_nih.%J.out

set -euo pipefail

SINGULARITY_DIR="${SAM3_SINGULARITY_DIR:-${SLURM_SUBMIT_DIR}}"
PROJECT_ROOT="${SAM3_PROJECT_ROOT:-$(cd "${SINGULARITY_DIR}/.." && pwd)}"
IMAGE_PATH="${SAM3_SINGULARITY_IMAGE:-${SINGULARITY_DIR}/sam3_master.simg}"
CHECKPOINT_PATH="${SAM3_CHECKPOINT:-${PROJECT_ROOT}/models/sam3/sam3.pt}"
DATA_ROOT="${SAM3_NIH_ROOT:-${PROJECT_ROOT}/../data/NIH_Dataset}"
BBOX_CSV="${SAM3_NIH_BBOX_CSV:-${PROJECT_ROOT}/Foundation & Open-Vocabulary Segmentation Models/BBox_List_2017.csv}"
OUTPUT_DIR="${SAM3_NIH_OUTPUT_DIR:-${PROJECT_ROOT}/sam3_outputs/nih_all_bbox_annotations}"

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
    echo "NIH-Bildordner fehlt: ${DATA_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${BBOX_CSV}" ]]; then
    echo "NIH-BBox-CSV fehlt: ${BBOX_CSV}" >&2
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
    --image-root "${DATA_ROOT}" \
    --bbox-csv "${BBOX_CSV}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --precision fp16 \
    --save-examples 0 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
