#!/usr/bin/env bash
set -euo pipefail

# Run this script from the directory containing the seg-project virtual environment.
# Usage:
#   COCO_ROOT=/data/coco ./run_experiment_venv.sh
#
# Optional environment variables: NUM_IMAGES, NUM_CLASSES, NUM_PERTURBATIONS,
# MAX_EDGE_SHIFT, SEED, OUTPUT, MODEL_ID, DEVICE.

COCO_ROOT="${COCO_ROOT:-./coco}"
NUM_IMAGES="${NUM_IMAGES:-100}"
NUM_CLASSES="${NUM_CLASSES:-10}"
NUM_PERTURBATIONS="${NUM_PERTURBATIONS:-10}"
MAX_EDGE_SHIFT="${MAX_EDGE_SHIFT:-0.05}"
SEED="${SEED:-20260713}"
OUTPUT="${OUTPUT:-./results/sam3_coco_box_robustness.jsonl}"
MODEL_ID="${MODEL_ID:-facebook/sam3}"
DEVICE="${DEVICE:-cuda}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "seg-project/bin/activate" ]]; then
  echo "Python environment activation script not found: seg-project/bin/activate" >&2
  echo "Run this script from the directory containing the seg-project environment." >&2
  exit 1
fi

# The virtual environment and all required packages are assumed to exist already.
# shellcheck disable=SC1091
source seg-project/bin/activate

if [[ ! -f "$COCO_ROOT/annotations/instances_val2017.json" ]]; then
  echo "Missing $COCO_ROOT/annotations/instances_val2017.json" >&2
  echo "Download COCO 2017 validation images and annotations, or set COCO_ROOT." >&2
  exit 1
fi

if [[ ! -d "$COCO_ROOT/val2017" ]]; then
  echo "Missing image directory: $COCO_ROOT/val2017" >&2
  exit 1
fi

python "$SCRIPT_DIR/run_sam3_coco_box_robustness.py" \
  --images-dir "$COCO_ROOT/val2017" \
  --annotations "$COCO_ROOT/annotations/instances_val2017.json" \
  --output "$OUTPUT" \
  --model-id "$MODEL_ID" \
  --num-images "$NUM_IMAGES" \
  --num-classes "$NUM_CLASSES" \
  --num-perturbations "$NUM_PERTURBATIONS" \
  --max-edge-shift "$MAX_EDGE_SHIFT" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --overwrite
