#!/usr/bin/env python3
"""Evaluate SAM3 robustness to perturbations of COCO ground-truth box prompts."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import Sam3TrackerModel, Sam3TrackerProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, required=True, help="COCO image directory, e.g. val2017")
    p.add_argument("--annotations", type=Path, required=True, help="COCO instances JSON")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    p.add_argument("--model-id", default="facebook/sam3", help="HF model ID or local checkpoint")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--num-perturbations", type=int, default=10)
    p.add_argument("--max-edge-shift", type=float, default=0.05,
                   help="Maximum absolute edge displacement as a fraction of box width/height")
    p.add_argument("--min-object-area", type=float, default=256.0,
                   help="Ignore COCO objects with segmentation area below this many pixels")
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Inference should already be deterministic for this model. This asks PyTorch
    # to use deterministic kernels where available without failing on unsupported ops.
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_objects(
    coco: COCO, num_images: int, num_classes: int, min_area: float, rng: random.Random
) -> list[dict[str, Any]]:
    """Choose unique images, approximately balanced over randomly selected classes."""
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.anns.values():
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        if ann.get("iscrowd", 0) or ann.get("area", 0) < min_area or w < 2 or h < 2:
            continue
        by_category[ann["category_id"]].append(ann)

    viable = [cid for cid, anns in by_category.items() if anns]
    if not viable:
        raise ValueError("No eligible non-crowd COCO annotations were found.")
    rng.shuffle(viable)
    chosen_categories = viable[: min(num_classes, len(viable))]
    for cid in chosen_categories:
        rng.shuffle(by_category[cid])

    selected: list[dict[str, Any]] = []
    used_images: set[int] = set()
    # Round-robin gives category coverage before adding a second example per class.
    positions = {cid: 0 for cid in chosen_categories}
    while len(selected) < num_images:
        progress = False
        for cid in chosen_categories:
            anns = by_category[cid]
            while positions[cid] < len(anns):
                ann = anns[positions[cid]]
                positions[cid] += 1
                if ann["image_id"] not in used_images:
                    selected.append(ann)
                    used_images.add(ann["image_id"])
                    progress = True
                    break
            if len(selected) == num_images:
                break
        if not progress:
            break

    if len(selected) < num_images:
        raise ValueError(
            f"Requested {num_images} unique images, but only {len(selected)} could be selected "
            f"from {len(chosen_categories)} sampled categories. Reduce --num-images, "
            "--min-object-area, or increase --num-classes."
        )
    rng.shuffle(selected)
    return selected


def coco_xywh_to_xyxy(box: list[float], image_w: int, image_h: int) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.array([x, y, min(x + w, image_w), min(y + h, image_h)], dtype=np.float32)


def perturb_box(
    box: np.ndarray, image_w: int, image_h: int, max_shift: float, rng: np.random.Generator
) -> tuple[np.ndarray, list[float]]:
    """Move each edge independently by U(-max_shift, max_shift) of box width/height."""
    x1, y1, x2, y2 = box.astype(float)
    w, h = x2 - x1, y2 - y1
    fractions = rng.uniform(-max_shift, max_shift, size=4)
    out = np.array(
        [x1 + fractions[0] * w, y1 + fractions[1] * h,
         x2 + fractions[2] * w, y2 + fractions[3] * h],
        dtype=np.float32,
    )
    out[[0, 2]] = np.clip(out[[0, 2]], 0, image_w)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, image_h)
    # Guard against a collapsed/inverted box after clipping or unusually large shifts.
    if out[2] <= out[0] + 1:
        out[2] = min(float(image_w), out[0] + 1)
        out[0] = max(0.0, out[2] - 1)
    if out[3] <= out[1] + 1:
        out[3] = min(float(image_h), out[1] + 1)
        out[1] = max(0.0, out[3] - 1)
    return out, fractions.tolist()


def mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = int(np.logical_and(pred, target).sum())
    pred_area, target_area = int(pred.sum()), int(target.sum())
    union = pred_area + target_area - intersection
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / (pred_area + target_area))
        if pred_area + target_area else 1.0,
        "predicted_area": pred_area,
        "ground_truth_area": target_area,
    }


def predict_boxes(
    model: Sam3TrackerModel,
    processor: Sam3TrackerProcessor,
    image: Image.Image,
    boxes: list[np.ndarray],
    device: torch.device,
) -> tuple[list[np.ndarray], list[float | None]]:
    # Shape: [image batch, objects, xyxy]. Each prompt is treated as one object.
    input_boxes = [[box.tolist() for box in boxes]]
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), inputs["original_sizes"], binarize=True
    )[0]
    # Expected [objects, 1, H, W]; tolerate a missing singleton mask-choice dimension.
    if masks.ndim == 4:
        masks = masks[:, 0]
    masks_np = [m.numpy().astype(bool) for m in masks]

    scores_tensor = getattr(outputs, "iou_scores", None)
    if scores_tensor is None:
        scores = [None] * len(masks_np)
    else:
        scores_arr = scores_tensor.detach().float().cpu().numpy()
        if scores_arr.ndim >= 3:
            scores_arr = scores_arr[..., 0]
        scores = [float(x) for x in np.ravel(scores_arr)[: len(masks_np)]]
    if len(masks_np) != len(boxes):
        raise RuntimeError(f"SAM3 returned {len(masks_np)} masks for {len(boxes)} box prompts")
    return masks_np, scores


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row["metrics"][key] for row in rows]))


def main() -> None:
    args = parse_args()
    if args.num_images < 1 or args.num_classes < 1 or args.num_perturbations < 1:
        raise ValueError("--num-images, --num-classes, and --num-perturbations must be positive")
    if not 0 <= args.max_edge_shift < 0.5:
        raise ValueError("--max-edge-shift must be in [0, 0.5)")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    seed_everything(args.seed)

    coco = COCO(str(args.annotations))
    annotations = select_objects(
        coco, args.num_images, args.num_classes, args.min_object_area, random.Random(args.seed)
    )
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU selected: using float32 instead of reduced precision.")
        dtype = torch.float32
    model = Sam3TrackerModel.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()
    processor = Sam3TrackerProcessor.from_pretrained(args.model_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np_rng = np.random.default_rng(args.seed)
    with args.output.open("w", encoding="utf-8") as fout:
        for ann in tqdm(annotations, desc="SAM3 inference"):
            image_info = coco.loadImgs([ann["image_id"]])[0]
            image_path = args.images_dir / image_info["file_name"]
            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size
            original_box = coco_xywh_to_xyxy(ann["bbox"], image_w, image_h)
            perturbed = [
                perturb_box(original_box, image_w, image_h, args.max_edge_shift, np_rng)
                for _ in range(args.num_perturbations)
            ]
            boxes = [original_box] + [item[0] for item in perturbed]
            predicted_masks, model_scores = predict_boxes(model, processor, image, boxes, device)
            gt_mask = coco.annToMask(ann).astype(bool)

            original = {
                "box_xyxy": original_box.tolist(),
                "model_iou_score": model_scores[0],
                "metrics": mask_metrics(predicted_masks[0], gt_mask),
            }
            perturbation_rows = []
            for i, ((box, edge_fractions), pred_mask, score) in enumerate(
                zip(perturbed, predicted_masks[1:], model_scores[1:])
            ):
                perturbation_rows.append({
                    "perturbation_index": i,
                    "box_xyxy": box.tolist(),
                    "edge_shift_fractions": {
                        "left": edge_fractions[0], "top": edge_fractions[1],
                        "right": edge_fractions[2], "bottom": edge_fractions[3],
                    },
                    "model_iou_score": score,
                    "metrics": mask_metrics(pred_mask, gt_mask),
                })

            category = coco.cats[ann["category_id"]]
            row = {
                "image_id": ann["image_id"],
                "file_name": image_info["file_name"],
                "annotation_id": ann["id"],
                "category_id": ann["category_id"],
                "category_name": category["name"],
                "image_size": {"width": image_w, "height": image_h},
                "ground_truth_box_xywh": [float(x) for x in ann["bbox"]],
                "original_prompt": original,
                "perturbed_prompt_average": {
                    "iou": mean_metric(perturbation_rows, "iou"),
                    "dice": mean_metric(perturbation_rows, "dice"),
                },
                "perturbed_prompts": perturbation_rows,
                "experiment": {
                    "seed": args.seed,
                    "model_id": args.model_id,
                    "max_edge_shift": args.max_edge_shift,
                    "num_perturbations": args.num_perturbations,
                },
            }
            fout.write(json.dumps(row, allow_nan=False) + "\n")
            fout.flush()

    print(f"Wrote {len(annotations)} image records to {args.output}")


if __name__ == "__main__":
    main()
