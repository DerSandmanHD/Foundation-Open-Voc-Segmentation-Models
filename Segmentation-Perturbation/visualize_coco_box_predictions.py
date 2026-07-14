#!/usr/bin/env python3
"""Render COCO ground truth and SAM2/SAM3 predictions for saved box prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from tqdm import tqdm


DEFAULT_MODEL_IDS = {
    "sam2": "facebook/sam2.1-hiera-large",
    "sam3": "facebook/sam3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=("auto", "sam2", "sam3"),
        default="auto",
        help="Use the JSONL metadata when set to auto (default)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the model ID recorded in the JSONL",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--box-width", type=int, default=4)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optionally visualize only the first N JSONL records",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path, max_images: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if max_images is not None and len(rows) >= max_images:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def resolve_model(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[str, str]:
    experiment = rows[0].get("experiment", {})
    family = args.model_family
    if family == "auto":
        family = experiment.get("model_family")
        if family not in ("sam2", "sam3"):
            recorded_id = str(experiment.get("model_id", "")).lower()
            family = "sam2" if "sam2" in recorded_id else "sam3"
    model_id = args.model_id or experiment.get("model_id") or DEFAULT_MODEL_IDS[family]
    return family, str(model_id)


def load_model(
    family: str, model_id: str, device: torch.device, dtype: torch.dtype
) -> tuple[Any, Any]:
    if family == "sam2":
        from transformers import Sam2Model, Sam2Processor

        model_class, processor_class = Sam2Model, Sam2Processor
    else:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        model_class, processor_class = Sam3TrackerModel, Sam3TrackerProcessor

    print(f"Loading {family.upper()} model: {model_id}")
    model = model_class.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    processor = processor_class.from_pretrained(model_id)
    return model, processor


def predict_boxes(
    model: Any,
    processor: Any,
    image: Image.Image,
    boxes: list[list[float]],
    device: torch.device,
) -> list[np.ndarray]:
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), inputs["original_sizes"], binarize=True
    )[0]
    if masks.ndim == 4:
        masks = masks[:, 0]
    result = [mask.numpy().astype(bool) for mask in masks]
    if len(result) != len(boxes):
        raise RuntimeError(f"Model returned {len(result)} masks for {len(boxes)} boxes")
    return result


def blend_mask(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    color_array = np.asarray(color, dtype=np.float32)
    output[mask] = (1.0 - alpha) * output[mask] + alpha * color_array
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def render_overlay(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    original_box: list[float],
    prompt_box: list[float],
    prompt_name: str,
    alpha: float,
    box_width: int,
) -> Image.Image:
    # Green = ground truth, red = prediction; their overlap appears yellowish.
    canvas = blend_mask(np.asarray(image), gt_mask, (0, 255, 0), alpha)
    canvas = blend_mask(canvas, pred_mask, (255, 0, 0), alpha)
    rendered = Image.fromarray(canvas)
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()

    # Always show the unperturbed ground-truth box as a cyan reference.
    draw.rectangle(original_box, outline=(0, 255, 255), width=box_width)
    if prompt_name != "original":
        # The orange rectangle is the actual perturbed box passed to the model.
        draw.rectangle(prompt_box, outline=(255, 165, 0), width=box_width)

    legend = [
        ((0, 255, 0), "Ground-truth mask"),
        ((255, 0, 0), "Predicted mask"),
        ((0, 255, 255), "Original box"),
    ]
    if prompt_name != "original":
        legend.append(((255, 165, 0), "Perturbed prompt box"))

    x, y = 10, 10
    for color, text in legend:
        draw.rectangle((x, y, x + 12, y + 12), fill=color, outline=(0, 0, 0))
        draw_label(draw, (x + 18, y), text, color, font)
        y += 19
    return rendered


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1")
    if args.box_width < 1:
        raise ValueError("--box-width must be positive")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive")

    rows = read_jsonl(args.results_jsonl, args.max_images)
    family, model_id = resolve_model(args, rows)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU selected: using float32 instead of reduced precision.")
        dtype = torch.float32
    model, processor = load_model(family, model_id, device, dtype)
    coco = COCO(str(args.annotations))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_index, row in enumerate(tqdm(rows, desc="Rendering images"), start=1):
        image_path = args.images_dir / row["file_name"]
        image = Image.open(image_path).convert("RGB")
        annotation_id = int(row["annotation_id"])
        if annotation_id not in coco.anns:
            raise KeyError(f"COCO annotation {annotation_id} not found")
        gt_mask = coco.annToMask(coco.anns[annotation_id]).astype(bool)

        original_box = [float(v) for v in row["original_prompt"]["box_xyxy"]]
        perturbation_rows = row.get("perturbed_prompts", [])
        perturbed_boxes = [
            [float(v) for v in perturbation["box_xyxy"]]
            for perturbation in perturbation_rows
        ]
        all_boxes = [original_box] + perturbed_boxes
        predicted_masks = predict_boxes(model, processor, image, all_boxes, device)

        image_output_dir = args.output_dir / f"img{image_index:03d}_{row['image_id']}"
        if image_output_dir.exists() and any(image_output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"{image_output_dir} is not empty; pass --overwrite to replace images"
            )
        image_output_dir.mkdir(parents=True, exist_ok=True)

        original_render = render_overlay(
            image, gt_mask, predicted_masks[0], original_box, original_box,
            "original", args.mask_alpha, args.box_width
        )
        original_render.save(image_output_dir / "original.png")

        for perturbation_index, (prompt_box, pred_mask) in enumerate(
            zip(perturbed_boxes, predicted_masks[1:]), start=1
        ):
            rendered = render_overlay(
                image, gt_mask, pred_mask, original_box, prompt_box,
                f"perturbed_{perturbation_index:02d}", args.mask_alpha, args.box_width
            )
            rendered.save(image_output_dir / f"perturbed_{perturbation_index:02d}.png")

    print(f"Wrote visualizations for {len(rows)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
