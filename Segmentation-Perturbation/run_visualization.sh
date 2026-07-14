#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing seg-project/bin/activate.
# Override any setting below with an environment variable.

COCO_ROOT="${COCO_ROOT:-./coco}"
RESULTS_JSONL="${RESULTS_JSONL:-./results/sam3_coco_box_robustness.jsonl}"
OUTPUT_IMAGES="${OUTPUT_IMAGES:-./output-images}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
MODEL_ID="${MODEL_ID:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MASK_ALPHA="${MASK_ALPHA:-0.45}"
BOX_WIDTH="${BOX_WIDTH:-4}"
MAX_IMAGES="${MAX_IMAGES:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "seg-project/bin/activate" ]]; then
  echo "Python environment not found: seg-project/bin/activate" >&2
  echo "Run this script from the directory containing seg-project." >&2
  exit 1
fi

# shellcheck disable=SC1091
source seg-project/bin/activate

ARGS=(
  --results-jsonl "$RESULTS_JSONL"
  --images-dir "$COCO_ROOT/val2017"
  --annotations "$COCO_ROOT/annotations/instances_val2017.json"
  --output-dir "$OUTPUT_IMAGES"
  --model-family "$MODEL_FAMILY"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --mask-alpha "$MASK_ALPHA"
  --box-width "$BOX_WIDTH"
  --overwrite
)

if [[ -n "$MODEL_ID" ]]; then
  ARGS+=(--model-id "$MODEL_ID")
fi
if [[ -n "$MAX_IMAGES" ]]; then
  ARGS+=(--max-images "$MAX_IMAGES")
fi

python "$SCRIPT_DIR/visualize_coco_box_predictions.py" "${ARGS[@]}"
