from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def add_sam_repo_to_path(sam_repo_root: Path) -> None:
    if not sam_repo_root.exists():
        raise FileNotFoundError(f"SAM repo not found: {sam_repo_root}")
    if str(sam_repo_root) not in sys.path:
        sys.path.insert(0, str(sam_repo_root))


def box_xywh_to_xyxy(box_xywh: List[float]) -> List[float]:
    x, y, w, h = box_xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def mask_to_bbox(mask_bool: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def evaluate_mask(mask_pred: np.ndarray, mask_gt: np.ndarray, gt_box: List[float]) -> Dict[str, object]:
    pred = mask_pred.astype(bool)
    gt = mask_gt.astype(bool)

    pred_area = int(pred.sum())
    gt_area = int(gt.sum())

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    x1, y1, x2, y2 = gt_box
    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(mask_pred.shape[1], round(x2)))
    y2 = int(min(mask_pred.shape[0], round(y2)))

    gt_box_mask = np.zeros_like(pred, dtype=bool)
    gt_box_mask[y1:y2, x1:x2] = True
    inside_gt_box = np.logical_and(pred, gt_box_mask).sum()

    pred_box = mask_to_bbox(pred)
    pred_box_iou = box_iou(pred_box, gt_box) if pred_box is not None else 0.0

    return {
        "mask_area": pred_area,
        "mask_iou_with_gt_mask": float(inter / union) if union > 0 else 0.0,
        "mask_precision_vs_gt_mask": float(inter / pred_area) if pred_area > 0 else 0.0,
        "mask_recall_vs_gt_mask": float(inter / gt_area) if gt_area > 0 else 0.0,
        "mask_inside_gt_box": float(inside_gt_box / pred_area) if pred_area > 0 else 0.0,
        "gt_box_covered_by_mask": float(inside_gt_box / gt_box_mask.sum()) if gt_box_mask.sum() > 0 else 0.0,
        "pred_box_iou_with_gt_box": float(pred_box_iou),
        "pred_box": pred_box,
    }


def get_foreground_point(mask_gt: np.ndarray, gt_box: List[float]) -> Tuple[float, float]:
    ys, xs = np.where(mask_gt.astype(bool))
    if len(xs) > 0 and len(ys) > 0:
        return float(xs.mean()), float(ys.mean())

    x1, y1, x2, y2 = gt_box
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def choose_best_mask(masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, int, float]:
    best_idx = int(np.argmax(scores))
    return masks[best_idx], best_idx, float(scores[best_idx])


def save_overlay(
    image: np.ndarray,
    gt_box: List[float],
    mask_pred: np.ndarray,
    out_path: Path,
    title: str,
    prompt_box: Optional[List[float]] = None,
    prompt_point: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)

    gt_rect = plt.Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(gt_rect)

    if prompt_box is not None:
        prompt_rect = plt.Rectangle(
            (prompt_box[0], prompt_box[1]),
            prompt_box[2] - prompt_box[0],
            prompt_box[3] - prompt_box[1],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(prompt_rect)

    if prompt_point is not None:
        ax.scatter([prompt_point[0]], [prompt_point[1]], s=80, marker="x", c="yellow")

    mask_show = np.where(mask_pred > 0.5, mask_pred, np.nan)
    ax.imshow(mask_show, cmap="cool", alpha=0.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def load_image(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def run_prompt_experiments(
    predictor,
    coco: COCO,
    image: np.ndarray,
    ann: Dict[str, object],
    category_name: str,
    file_name: str,
) -> List[Dict[str, object]]:
    gt_box = box_xywh_to_xyxy(ann["bbox"])
    mask_gt = coco.annToMask(ann)
    point = get_foreground_point(mask_gt, gt_box)

    predictor.set_image(image)

    box_masks, box_scores, _ = predictor.predict(
        box=np.array(gt_box, dtype=np.float32),
        multimask_output=True,
        return_logits=False,
    )
    box_mask, box_best_idx, box_best_score = choose_best_mask(box_masks, box_scores)
    box_metrics = evaluate_mask(box_mask, mask_gt, gt_box)

    point_masks, point_scores, _ = predictor.predict(
        point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
        return_logits=False,
    )
    point_mask, point_best_idx, point_best_score = choose_best_mask(point_masks, point_scores)
    point_metrics = evaluate_mask(point_mask, mask_gt, gt_box)

    base = {
        "image_id": int(ann["image_id"]),
        "annotation_id": int(ann["id"]),
        "category_id": int(ann["category_id"]),
        "category_name": category_name,
        "image_file": file_name,
        "gt_x1": gt_box[0],
        "gt_y1": gt_box[1],
        "gt_x2": gt_box[2],
        "gt_y2": gt_box[3],
        "point_x": point[0],
        "point_y": point[1],
        "gt_mask_area": int(mask_gt.sum()),
    }

    return [
        {
            **base,
            "status": "ok",
            "experiment": "sam_box_prompt",
            "prompt_type": "box",
            "sam_score": box_best_score,
            "best_mask_index": box_best_idx,
            **box_metrics,
            "_mask_pred": box_mask,
            "_prompt_box": gt_box,
            "_prompt_point": None,
        },
        {
            **base,
            "status": "ok",
            "experiment": "sam_point_prompt",
            "prompt_type": "point",
            "sam_score": point_best_score,
            "best_mask_index": point_best_idx,
            **point_metrics,
            "_mask_pred": point_mask,
            "_prompt_box": None,
            "_prompt_point": point,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM on a COCO subset using box and point prompts.")
    parser.add_argument("--coco_images", type=str, required=True, help="Path to the COCO image folder, e.g. val2017.")
    parser.add_argument("--coco_annotations", type=str, required=True, help="Path to instances_val2017.json.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the SAM checkpoint .pth file.")
    parser.add_argument(
        "--sam_repo_root",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "SAM Repo" / "segment-anything").resolve()),
        help="Path to the local segment-anything repository.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type. vit_b is the lightest option for a laptop.",
    )
    parser.add_argument("--output_dir", type=str, default=str((Path(__file__).resolve().parent / "results").resolve()))
    parser.add_argument("--max_annotations", type=int, default=50, help="Maximum number of annotations to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling annotations.")
    parser.add_argument("--save_examples", type=int, default=10, help="How many visual examples to save per experiment.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    args = parse_args()

    coco_images = Path(args.coco_images)
    coco_annotations = Path(args.coco_annotations)
    checkpoint = Path(args.checkpoint)
    sam_repo_root = Path(args.sam_repo_root)
    output_dir = Path(args.output_dir)

    if not coco_images.exists():
        raise FileNotFoundError(f"COCO image folder not found: {coco_images}")
    if not coco_annotations.exists():
        raise FileNotFoundError(f"COCO annotations file not found: {coco_annotations}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    add_sam_repo_to_path(sam_repo_root)

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    coco = COCO(str(coco_annotations))
    annotations = coco.loadAnns(coco.getAnnIds(iscrowd=False))

    if len(annotations) == 0:
        raise RuntimeError("No annotations found in the supplied COCO file.")

    random.seed(args.seed)
    random.shuffle(annotations)
    annotations = annotations[: max(1, args.max_annotations)]

    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    category_lookup = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

    rows: List[Dict[str, object]] = []
    saved_examples = {"sam_box_prompt": 0, "sam_point_prompt": 0}

    for ann in tqdm(annotations, total=len(annotations), desc="Evaluating"):
        image_info = coco.loadImgs([ann["image_id"]])[0]
        image_path = coco_images / image_info["file_name"]
        category_name = category_lookup.get(int(ann["category_id"]), "unknown")

        if not image_path.exists():
            rows.append(
                {
                    "image_id": int(ann["image_id"]),
                    "annotation_id": int(ann["id"]),
                    "category_id": int(ann["category_id"]),
                    "category_name": category_name,
                    "image_file": image_info["file_name"],
                    "status": "image_not_found",
                }
            )
            continue

        image = load_image(image_path)

        try:
            rows_for_ann = run_prompt_experiments(
                predictor=predictor,
                coco=coco,
                image=image,
                ann=ann,
                category_name=category_name,
                file_name=image_info["file_name"],
            )
            rows.extend(rows_for_ann)

            if saved_examples["sam_box_prompt"] < args.save_examples:
                box_row = rows_for_ann[0]
                save_overlay(
                    image=image,
                    gt_box=[box_row["gt_x1"], box_row["gt_y1"], box_row["gt_x2"], box_row["gt_y2"]],
                    mask_pred=box_row["_mask_pred"],
                    out_path=examples_dir / f"{int(ann['image_id'])}_{int(ann['id'])}_box.png",
                    title=f"SAM box prompt | {category_name}",
                    prompt_box=box_row["_prompt_box"],
                )
                saved_examples["sam_box_prompt"] += 1

            if saved_examples["sam_point_prompt"] < args.save_examples:
                point_row = rows_for_ann[1]
                save_overlay(
                    image=image,
                    gt_box=[point_row["gt_x1"], point_row["gt_y1"], point_row["gt_x2"], point_row["gt_y2"]],
                    mask_pred=point_row["_mask_pred"],
                    out_path=examples_dir / f"{int(ann['image_id'])}_{int(ann['id'])}_point.png",
                    title=f"SAM point prompt | {category_name}",
                    prompt_point=point_row["_prompt_point"],
                )
                saved_examples["sam_point_prompt"] += 1

        except Exception as exc:
            rows.append(
                {
                    "image_id": int(ann["image_id"]),
                    "annotation_id": int(ann["id"]),
                    "category_id": int(ann["category_id"]),
                    "category_name": category_name,
                    "image_file": image_info["file_name"],
                    "status": "error",
                    "error": str(exc),
                }
            )

    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    result_df = pd.DataFrame(clean_rows)
    results_csv = output_dir / "sam_coco_results.csv"
    result_df.to_csv(results_csv, index=False)

    print(f"Saved detailed results to: {results_csv}")
    print(f"Saved example overlays to: {examples_dir}")

    if "status" in result_df.columns:
        ok_df = result_df[result_df["status"] == "ok"].copy()
    else:
        ok_df = result_df.copy()

    if len(ok_df) > 0:
        metric_cols = [
            "sam_score",
            "mask_iou_with_gt_mask",
            "mask_precision_vs_gt_mask",
            "mask_recall_vs_gt_mask",
            "mask_inside_gt_box",
            "gt_box_covered_by_mask",
            "pred_box_iou_with_gt_box",
            "mask_area",
        ]
        available_metrics = [col for col in metric_cols if col in ok_df.columns]

        summary_mean = ok_df.groupby("experiment")[available_metrics].mean()
        summary_median = ok_df.groupby("experiment")[available_metrics].median()

        summary_mean_csv = output_dir / "sam_coco_summary_mean.csv"
        summary_median_csv = output_dir / "sam_coco_summary_median.csv"
        summary_mean.to_csv(summary_mean_csv)
        summary_median.to_csv(summary_median_csv)

        print("\nMean summary:\n", summary_mean)
        print("\nMedian summary:\n", summary_median)
        print(f"Saved mean summary to: {summary_mean_csv}")
        print(f"Saved median summary to: {summary_median_csv}")


if __name__ == "__main__":
    main()