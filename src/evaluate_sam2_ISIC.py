from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Geometrie & Metriken
# ---------------------------------------------------------------------------

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


def gt_mask_to_bbox(mask_gt: np.ndarray) -> Optional[List[float]]:
    """Bounding Box aus GT-Maske ableiten (für Box-Prompt)."""
    bbox = mask_to_bbox(mask_gt.astype(bool))
    if bbox is None:
        return None
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


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
        "gt_mask_area": gt_area,
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
    mask_gt: np.ndarray,
    out_path: Path,
    title: str,
    prompt_box: Optional[List[float]] = None,
    prompt_point: Optional[Tuple[float, float]] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, mask, label in zip(axes, [mask_gt, mask_pred], ["Ground Truth", "SAM2 Prediction"]):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"{title}\n{label}")

        gt_rect = plt.Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
        )
        ax.add_patch(gt_rect)

        if prompt_box is not None and label == "SAM2 Prediction":
            pr = plt.Rectangle(
                (prompt_box[0], prompt_box[1]),
                prompt_box[2] - prompt_box[0],
                prompt_box[3] - prompt_box[1],
                linewidth=2, edgecolor="yellow", facecolor="none",
            )
            ax.add_patch(pr)

        if prompt_point is not None and label == "SAM2 Prediction":
            ax.scatter([prompt_point[0]], [prompt_point[1]], s=80, marker="x", c="yellow")

        mask_show = np.where(mask.astype(bool), 1.0, np.nan)
        ax.imshow(mask_show, cmap="cool", alpha=0.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ISIC Datensatz laden
# ---------------------------------------------------------------------------

def load_isic_pairs(images_dir: Path, gt_dir: Path) -> List[Dict[str, Path]]:
    """
    Findet alle Bild/Masken-Paare im ISIC-Format.
    Bilder:  ISIC_XXXXXXX.jpg
    Masken:  ISIC_XXXXXXX_segmentation.png
    """
    pairs = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem  # z.B. ISIC_0012345
        mask_path = gt_dir / f"{stem}_segmentation.png"
        if mask_path.exists():
            pairs.append({"image": img_path, "mask": mask_path, "id": stem})
        else:
            print(f"[WARN] Keine Maske gefunden für {img_path.name}, überspringe.")
    return pairs


# ---------------------------------------------------------------------------
# SAM2 Inferenz
# ---------------------------------------------------------------------------

def run_prompt_experiments_sam2(
    predictor,
    image: np.ndarray,
    mask_gt: np.ndarray,
    image_id: str,
) -> List[Dict[str, object]]:
    import torch

    gt_box = gt_mask_to_bbox(mask_gt)
    if gt_box is None:
        return []

    point = get_foreground_point(mask_gt, gt_box)
    predictor.set_image(image)

    # --- Box Prompt ---
    with torch.inference_mode():
        box_masks, box_scores, _ = predictor.predict(
            box=np.array(gt_box, dtype=np.float32),
            multimask_output=True,
        )
    best_idx = int(np.argmax(box_scores))
    box_mask = box_masks[best_idx].astype(np.uint8)
    box_metrics = evaluate_mask(box_mask, mask_gt, gt_box)

    # --- Point Prompt ---
    with torch.inference_mode():
        point_masks, point_scores, _ = predictor.predict(
            point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
    best_idx_p = int(np.argmax(point_scores))
    point_mask = point_masks[best_idx_p].astype(np.uint8)
    point_metrics = evaluate_mask(point_mask, mask_gt, gt_box)

    base = {
        "image_id": image_id,
        "gt_x1": gt_box[0], "gt_y1": gt_box[1],
        "gt_x2": gt_box[2], "gt_y2": gt_box[3],
        "point_x": point[0], "point_y": point[1],
    }

    return [
        {
            **base,
            "status": "ok",
            "experiment": "sam2_box_prompt",
            "prompt_type": "box",
            "sam_score": float(box_scores[best_idx]),
            "best_mask_index": best_idx,
            **box_metrics,
            "_mask_pred": box_mask,
            "_mask_gt": mask_gt,
            "_prompt_box": gt_box,
            "_prompt_point": None,
        },
        {
            **base,
            "status": "ok",
            "experiment": "sam2_point_prompt",
            "prompt_type": "point",
            "sam_score": float(point_scores[best_idx_p]),
            "best_mask_index": best_idx_p,
            **point_metrics,
            "_mask_pred": point_mask,
            "_mask_gt": mask_gt,
            "_prompt_box": None,
            "_prompt_point": point,
        },
    ]


# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM 2.1 on ISIC 2018 dataset.")
    parser.add_argument("--isic_images", type=str, required=True,
                        help="Pfad zum ISIC Bildordner (z.B. data/ISIC2018/Validation_Input).")
    parser.add_argument("--isic_masks", type=str, required=True,
                        help="Pfad zum ISIC Maskenordner (z.B. data/ISIC2018/Validation_GroundTruth).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pfad zum SAM 2.1 Checkpoint (.pt Datei).")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM2 config YAML (muss zur Checkpoint-Größe passen).")
    parser.add_argument("--output_dir", type=str, default="./results_sam2_isic",
                        help="Ausgabeordner für Ergebnisse und Plots.")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximale Anzahl Bilder. Wenn nicht gesetzt: alle.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_examples", type=int, default=10,
                        help="Wie viele Beispielbilder gespeichert werden sollen.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
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

    isic_images = Path(args.isic_images)
    isic_masks = Path(args.isic_masks)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not isic_images.exists():
        raise FileNotFoundError(f"Bildordner nicht gefunden: {isic_images}")
    if not isic_masks.exists():
        raise FileNotFoundError(f"Maskenordner nicht gefunden: {isic_masks}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # SAM2 laden
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "sam2 package nicht gefunden. Installieren mit:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        )

    print(f"Lade SAM2 Modell: {checkpoint.name}")
    sam2_model = build_sam2(args.model_cfg, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Modell erfolgreich geladen.")

    # Dateipaare laden
    pairs = load_isic_pairs(isic_images, isic_masks)
    if len(pairs) == 0:
        raise RuntimeError(f"Keine Bild/Masken-Paare in {isic_images} / {isic_masks} gefunden.")

    random.seed(args.seed)
    random.shuffle(pairs)

    if args.max_images is not None:
        pairs = pairs[:max(1, args.max_images)]
        print(f"Evaluiere {len(pairs)} Bilder (begrenzt durch --max_images).")
    else:
        print(f"Evaluiere alle {len(pairs)} Bilder.")

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    saved_examples = {"sam2_box_prompt": 0, "sam2_point_prompt": 0}

    for pair in tqdm(pairs, desc="Evaluating ISIC"):
        image_np = np.array(Image.open(pair["image"]).convert("RGB"))
        mask_gt = (np.array(Image.open(pair["mask"]).convert("L")) > 127).astype(np.uint8)

        try:
            rows_for_img = run_prompt_experiments_sam2(
                predictor=predictor,
                image=image_np,
                mask_gt=mask_gt,
                image_id=pair["id"],
            )
            rows.extend(rows_for_img)

            if rows_for_img and saved_examples["sam2_box_prompt"] < args.save_examples:
                box_row = rows_for_img[0]
                save_overlay(
                    image=image_np,
                    gt_box=[box_row["gt_x1"], box_row["gt_y1"], box_row["gt_x2"], box_row["gt_y2"]],
                    mask_pred=box_row["_mask_pred"],
                    mask_gt=box_row["_mask_gt"],
                    out_path=examples_dir / f"{pair['id']}_box.png",
                    title=f"SAM2 Box | {pair['id']}",
                    prompt_box=box_row["_prompt_box"],
                )
                saved_examples["sam2_box_prompt"] += 1

            if rows_for_img and saved_examples["sam2_point_prompt"] < args.save_examples:
                point_row = rows_for_img[1]
                save_overlay(
                    image=image_np,
                    gt_box=[point_row["gt_x1"], point_row["gt_y1"], point_row["gt_x2"], point_row["gt_y2"]],
                    mask_pred=point_row["_mask_pred"],
                    mask_gt=point_row["_mask_gt"],
                    out_path=examples_dir / f"{pair['id']}_point.png",
                    title=f"SAM2 Point | {pair['id']}",
                    prompt_point=point_row["_prompt_point"],
                )
                saved_examples["sam2_point_prompt"] += 1

        except Exception as exc:
            rows.append({
                "image_id": pair["id"],
                "status": "error",
                "error": str(exc),
            })

    # Ergebnisse speichern
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    result_df = pd.DataFrame(clean_rows)
    results_csv = output_dir / "sam2_isic_results.csv"
    result_df.to_csv(results_csv, index=False)
    print(f"\nErgebnisse gespeichert: {results_csv}")
    print(f"Beispielbilder gespeichert: {examples_dir}")

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

        summary_mean.to_csv(output_dir / "sam2_isic_summary_mean.csv")
        summary_median.to_csv(output_dir / "sam2_isic_summary_median.csv")

        print("\nMean summary:\n", summary_mean)
        print("\nMedian summary:\n", summary_median)


if __name__ == "__main__":
    main()