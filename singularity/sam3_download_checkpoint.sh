#!/usr/bin/env bash
set -euo pipefail

SINGULARITY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PATH="${SAM3_SINGULARITY_IMAGE:-${SINGULARITY_DIR}/sam3_master.simg}"
MODEL_DIR="${SAM3_MODEL_DIR:-${SINGULARITY_DIR}/models}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Fehler: Singularity-Image nicht gefunden: ${IMAGE_PATH}" >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}"

singularity exec "${IMAGE_PATH}" hf download facebook/sam3 \
    sam3.pt config.json \
    --local-dir "${MODEL_DIR}"

echo "Checkpoint: ${MODEL_DIR}/sam3.pt"
