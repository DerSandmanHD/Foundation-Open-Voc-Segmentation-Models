from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sam3_cluster.common import (
    cuda_inference_context,
    load_image_model,
    select_highest_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 text-prompt smoke test")
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument("--prompt", default="lungs", help="Short text concept")
    parser.add_argument("--checkpoint", default=None, help="Optional local sam3.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output-dir", default="sam3_outputs/smoke_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    _, processor, device = load_image_model(args.checkpoint, args.threshold)
    with cuda_inference_context(args.precision):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)
    selected = select_highest_score(output)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].axis("off")

    result = {
        "image": str(image_path),
        "prompt": args.prompt,
        "threshold": args.threshold,
        "device": device,
        "precision": args.precision,
        "num_detections": 0,
    }
    if selected is None:
        axes[1].set_title(f"Keine Detektion für '{args.prompt}'")
    else:
        axes[1].imshow(selected["mask"], cmap="jet", alpha=0.45)
        x1, y1, x2, y2 = selected["box"]
        axes[1].add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2)
        )
        axes[1].set_title(f"{args.prompt} | Score {selected['score']:.3f}")
        result.update(
            num_detections=selected["num_detections"],
            selected_score=selected["score"],
            selected_box=selected["box"],
        )

    fig.tight_layout()
    fig.savefig(output_dir / "result.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
