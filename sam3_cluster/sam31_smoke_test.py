from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam3_cluster.common import require_complete_sam3_install, require_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3.1 text-prompt smoke test")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument("--prompt", default="truck", help="Short text concept")
    parser.add_argument("--checkpoint", required=True, help="Local sam3.1_multiplex.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="sam3_outputs/smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")

    require_complete_sam3_install()
    require_cuda()

    from sam3 import build_sam3_predictor

    print("Lade SAM3.1 Multiplex-Predictor ...", flush=True)
    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint_path),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=False,
        async_loading_frames=False,
        default_output_prob_thresh=args.threshold,
    )

    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(image_path)}
    )
    session_id = response["session_id"]
    try:
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": args.prompt,
                "output_prob_thresh": args.threshold,
            }
        )
        outputs = response["outputs"]
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )

    probabilities = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
    boxes_xywh = np.asarray(outputs.get("out_boxes_xywh", []), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": "sam3.1",
        "checkpoint": str(checkpoint_path),
        "image": str(image_path),
        "prompt": args.prompt,
        "threshold": args.threshold,
        "num_detections": int(len(probabilities)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].axis("off")

    if len(probabilities) == 0:
        axes[1].set_title(f"Keine Detektion für '{args.prompt}'")
    else:
        best_index = int(np.argmax(probabilities))
        best_mask = np.squeeze(masks[best_index])
        x, y, box_width, box_height = boxes_xywh[best_index]
        box_xyxy = [
            float(x * width),
            float(y * height),
            float((x + box_width) * width),
            float((y + box_height) * height),
        ]
        x1, y1, x2, y2 = box_xyxy
        axes[1].imshow(best_mask, cmap="jet", alpha=0.45)
        axes[1].add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color="lime", linewidth=2,
            )
        )
        axes[1].set_title(
            f"SAM3.1: {args.prompt} | Score {probabilities[best_index]:.3f}"
        )
        result.update(
            selected_score=float(probabilities[best_index]),
            selected_box_xyxy=box_xyxy,
            selected_mask_area=int(best_mask.sum()),
        )

    fig.tight_layout()
    fig.savefig(output_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
