from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Verhindert Fehler auf Headless-Servern

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Geometrie-Hilfsfunktionen (identisch zu SAM1-Skript)
# ---------------------------------------------------------------------------

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

    dice = (2.0 * inter) / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 0.0

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
        "dice_score": float(dice),
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


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

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
        linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
    )
    ax.add_patch(gt_rect)

    if prompt_box is not None:
        prompt_rect = plt.Rectangle(
            (prompt_box[0], prompt_box[1]),
            prompt_box[2] - prompt_box[0],
            prompt_box[3] - prompt_box[1],
            linewidth=2, edgecolor="yellow", facecolor="none",
        )
        ax.add_patch(prompt_rect)

    if prompt_point is not None:
        ax.scatter([prompt_point[0]], [prompt_point[1]], s=80, marker="x", c="yellow")

    mask_show = np.where(mask_pred > 0.5, mask_pred.astype(float), np.nan)
    ax.imshow(mask_show, cmap="cool", alpha=0.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def load_image(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


# ---------------------------------------------------------------------------
# SAM 2.1 Inferenz
# ---------------------------------------------------------------------------

def run_prompt_experiments_sam2(
    predictor,
    coco: COCO,
    image: np.ndarray,
    ann: Dict[str, object],
    category_name: str,
    file_name: str,
) -> List[Dict[str, object]]:
    import torch

    gt_box = box_xywh_to_xyxy(ann["bbox"])
    mask_gt = coco.annToMask(ann)
    point = get_foreground_point(mask_gt, gt_box)

    # SAM2 erwartet set_image mit numpy array (H, W, 3) uint8
    predictor.set_image(image)

    # --- Box Prompt ---
    box_np = np.array(gt_box, dtype=np.float32)
    with torch.inference_mode():
        box_masks, box_scores, _ = predictor.predict(
            box=box_np,
            multimask_output=True,
        )
    # box_masks: (N, H, W) bool
    best_idx = int(np.argmax(box_scores))
    box_mask = box_masks[best_idx].astype(np.uint8)
    box_best_score = float(box_scores[best_idx])
    box_metrics = evaluate_mask(box_mask, mask_gt, gt_box)

    # --- Point Prompt ---
    point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    with torch.inference_mode():
        point_masks, point_scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    best_idx_p = int(np.argmax(point_scores))
    point_mask = point_masks[best_idx_p].astype(np.uint8)
    point_best_score = float(point_scores[best_idx_p])
    point_metrics = evaluate_mask(point_mask, mask_gt, gt_box)

    base = {
        "image_id": int(ann["image_id"]),
        "annotation_id": int(ann["id"]),
        "category_id": int(ann["category_id"]),
        "category_name": category_name,
        "image_file": file_name,
        "gt_x1": gt_box[0], "gt_y1": gt_box[1],
        "gt_x2": gt_box[2], "gt_y2": gt_box[3],
        "point_x": point[0], "point_y": point[1],
        "gt_mask_area": int(mask_gt.sum()),
    }

    return [
        {
            **base,
            "status": "ok",
            "experiment": "sam2_box_prompt",
            "prompt_type": "box",
            "sam_score": box_best_score,
            "best_mask_index": best_idx,
            **box_metrics,
            "_mask_pred": box_mask,
            "_prompt_box": gt_box,
            "_prompt_point": None,
        },
        {
            **base,
            "status": "ok",
            "experiment": "sam2_point_prompt",
            "prompt_type": "point",
            "sam_score": point_best_score,
            "best_mask_index": best_idx_p,
            **point_metrics,
            "_mask_pred": point_mask,
            "_prompt_box": None,
            "_prompt_point": point,
        },
    ]


# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM 2.1 on a COCO subset using box and point prompts.")
    parser.add_argument("--coco_images", type=str, required=True, help="Path to the COCO image folder, e.g. val2017.")
    parser.add_argument("--coco_annotations", type=str, required=True, help="Path to instances_val2017.json.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the SAM 2.1 checkpoint .pt file.")
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help=(
            "SAM2 config YAML. Common options:\n"
            "  configs/sam2.1/sam2.1_hiera_t.yaml  (tiny)\n"
            "  configs/sam2.1/sam2.1_hiera_s.yaml  (small)\n"
            "  configs/sam2.1/sam2.1_hiera_b+.yaml (base+)\n"
            "  configs/sam2.1/sam2.1_hiera_l.yaml  (large)"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./results_sam2", help="Directory to save results and plots.")
    parser.add_argument("--max_annotations", type=int, default=None, help="Maximum number of annotations to evaluate. If None, use all.")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    coco_images = Path(args.coco_images)
    coco_annotations = Path(args.coco_annotations)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not coco_images.exists():
        raise FileNotFoundError(f"COCO image folder not found: {coco_images}")
    if not coco_annotations.exists():
        raise FileNotFoundError(f"COCO annotations file not found: {coco_annotations}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # SAM 2.1 laden
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "sam2 package not found. Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )

    print(f"Loading SAM2 model: {checkpoint.name}")
    print(f"Using config: {args.model_cfg}")
    sam2_model = build_sam2(args.model_cfg, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Model loaded successfully.")

    # COCO laden
    coco = COCO(str(coco_annotations))
    annotations = coco.loadAnns(coco.getAnnIds(iscrowd=False))

    if len(annotations) == 0:
        raise RuntimeError("No annotations found in the supplied COCO file.")

    random.seed(args.seed)
    random.shuffle(annotations)

    # ============================================================
    # HIER IST DIE ÄNDERUNG: Wenn max_annotations None ist, nimm alle
    # ============================================================
    if args.max_annotations is not None:
        annotations = annotations[:max(1, args.max_annotations)]
        print(f"Evaluating {len(annotations)} annotations (limited by --max_annotations).")
    else:
        print(f"Evaluating all {len(annotations)} annotations (no limit set).")

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    category_lookup = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

    rows: List[Dict[str, object]] = []
    saved_examples = {"sam2_box_prompt": 0, "sam2_point_prompt": 0}

    for ann in tqdm(annotations, total=len(annotations), desc="Evaluating"):
        image_info = coco.loadImgs([ann["image_id"]])[0]
        image_path = coco_images / image_info["file_name"]
        category_name = category_lookup.get(int(ann["category_id"]), "unknown")

        if not image_path.exists():
            rows.append({
                "image_id": int(ann["image_id"]),
                "annotation_id": int(ann["id"]),
                "category_id": int(ann["category_id"]),
                "category_name": category_name,
                "image_file": image_info["file_name"],
                "status": "image_not_found",
            })
            continue

        image = load_image(image_path)

        try:
            rows_for_ann = run_prompt_experiments_sam2(
                predictor=predictor,
                coco=coco,
                image=image,
                ann=ann,
                category_name=category_name,
                file_name=image_info["file_name"],
            )
            rows.extend(rows_for_ann)

            if saved_examples["sam2_box_prompt"] < args.save_examples:
                box_row = rows_for_ann[0]
                save_overlay(
                    image=image,
                    gt_box=[box_row["gt_x1"], box_row["gt_y1"], box_row["gt_x2"], box_row["gt_y2"]],
                    mask_pred=box_row["_mask_pred"],
                    out_path=examples_dir / f"{int(ann['image_id'])}_{int(ann['id'])}_box.png",
                    title=f"SAM2 box prompt | {category_name}",
                    prompt_box=box_row["_prompt_box"],
                )
                saved_examples["sam2_box_prompt"] += 1

            if saved_examples["sam2_point_prompt"] < args.save_examples:
                point_row = rows_for_ann[1]
                save_overlay(
                    image=image,
                    gt_box=[point_row["gt_x1"], point_row["gt_y1"], point_row["gt_x2"], point_row["gt_y2"]],
                    mask_pred=point_row["_mask_pred"],
                    out_path=examples_dir / f"{int(ann['image_id'])}_{int(ann['id'])}_point.png",
                    title=f"SAM2 point prompt | {category_name}",
                    prompt_point=point_row["_prompt_point"],
                )
                saved_examples["sam2_point_prompt"] += 1

        except Exception as exc:
            rows.append({
                "image_id": int(ann["image_id"]),
                "annotation_id": int(ann["id"]),
                "category_id": int(ann["category_id"]),
                "category_name": category_name,
                "image_file": image_info["file_name"],
                "status": "error",
                "error": str(exc),
            })

    # Ergebnisse speichern
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    result_df = pd.DataFrame(clean_rows)
    results_csv = output_dir / "sam2_coco_results.csv"
    result_df.to_csv(results_csv, index=False)

    print(f"\nSaved detailed results to: {results_csv}")
    print(f"Saved example overlays to: {examples_dir}")

    ok_df = result_df[result_df["status"] == "ok"].copy() if "status" in result_df.columns else result_df.copy()

    if len(ok_df) > 0:
        metric_cols = [
            "sam_score",
            "dice_score",
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

        summary_mean.to_csv(output_dir / "sam2_coco_summary_mean.csv")
        summary_median.to_csv(output_dir / "sam2_coco_summary_median.csv")

        print("\nMean summary:\n", summary_mean)
        print("\nMedian summary:\n", summary_median)


if __name__ == "__main__":
    main()