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
# BraTS H5 laden & zu RGB konvertieren
# ---------------------------------------------------------------------------

def load_brats_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lädt ein BraTS H5-Slice.
    image: (240,240,4) → FLAIR-Kanal (Index 3) → normalisiert → RGB uint8
    mask:  (240,240,3) → alle 3 Tumor-Klassen kombiniert → binäre Maske
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        image_raw = f["image"][:]
        mask_raw = f["mask"][:]

    flair = image_raw[:, :, 3].astype(np.float32)
    flair_min, flair_max = flair.min(), flair.max()
    if flair_max > flair_min:
        flair_norm = ((flair - flair_min) / (flair_max - flair_min) * 255).astype(np.uint8)
    else:
        flair_norm = np.zeros_like(flair, dtype=np.uint8)

    image_rgb = np.stack([flair_norm, flair_norm, flair_norm], axis=-1)
    mask_gt = (mask_raw.sum(axis=-1) > 0).astype(np.uint8)

    return image_rgb, mask_gt


def load_brats_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("volume_*_slice_*.h5"))


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

    for ax, mask, label in zip(axes, [mask_gt, mask_pred], ["Ground Truth", "Prediction"]):
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{title}\n{label}")

        gt_rect = plt.Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
        )
        ax.add_patch(gt_rect)

        if prompt_box is not None and label == "Prediction":
            pr = plt.Rectangle(
                (prompt_box[0], prompt_box[1]),
                prompt_box[2] - prompt_box[0],
                prompt_box[3] - prompt_box[1],
                linewidth=2, edgecolor="yellow", facecolor="none",
            )
            ax.add_patch(pr)

        if prompt_point is not None and label == "Prediction":
            ax.scatter([prompt_point[0]], [prompt_point[1]], s=80, marker="x", c="yellow")

        mask_show = np.where(mask.astype(bool), 1.0, np.nan)
        ax.imshow(mask_show, cmap="cool", alpha=0.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SAM2 Inferenz
# ---------------------------------------------------------------------------

def run_prompt_experiments(
    predictor,
    image: np.ndarray,
    mask_gt: np.ndarray,
    slice_id: str,
    model_name: str,
) -> List[Dict[str, object]]:
    import torch

    gt_box = gt_mask_to_bbox(mask_gt)
    if gt_box is None:
        return []

    point = get_foreground_point(mask_gt, gt_box)
    predictor.set_image(image)

    with torch.inference_mode():
        box_masks, box_scores, _ = predictor.predict(
            box=np.array(gt_box, dtype=np.float32),
            multimask_output=True,
        )
    best_idx = int(np.argmax(box_scores))
    box_mask = box_masks[best_idx].astype(np.uint8)
    box_metrics = evaluate_mask(box_mask, mask_gt, gt_box)

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
        "slice_id": slice_id,
        "model": model_name,
        "gt_x1": gt_box[0], "gt_y1": gt_box[1],
        "gt_x2": gt_box[2], "gt_y2": gt_box[3],
        "point_x": point[0], "point_y": point[1],
    }

    return [
        {
            **base,
            "status": "ok",
            "experiment": f"{model_name}_box_prompt",
            "prompt_type": "box",
            "sam_score": float(box_scores[best_idx]),
            **box_metrics,
            "_mask_pred": box_mask,
            "_mask_gt": mask_gt,
            "_prompt_box": gt_box,
            "_prompt_point": None,
        },
        {
            **base,
            "status": "ok",
            "experiment": f"{model_name}_point_prompt",
            "prompt_type": "point",
            "sam_score": float(point_scores[best_idx_p]),
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
    parser = argparse.ArgumentParser(description="Evaluate SAM2 / MedSAM2 on BraTS 2020.")
    parser.add_argument("--brats_data", type=str, required=True,
                        help="Pfad zum BraTS H5-Datenordner.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pfad zum Checkpoint (.pt Datei).")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM2 config YAML. Optionen: hiera_t, hiera_s, hiera_b+, hiera_l")
    parser.add_argument("--model_name", type=str, default="sam2",
                        help="Name des Modells für die Ergebnisse, z.B. 'sam2_large', 'medsam2'.")
    parser.add_argument("--output_dir", type=str, default="./results_brats",
                        help="Ausgabeordner für Ergebnisse.")
    parser.add_argument("--max_slices", type=int, default=None,
                        help="Maximale Anzahl Slices. Wenn nicht gesetzt: alle.")
    parser.add_argument("--only_tumor", action="store_true", default=True,
                        help="Nur Slices mit Tumor evaluieren (empfohlen).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_examples", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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

    brats_data = Path(args.brats_data)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not brats_data.exists():
        raise FileNotFoundError(f"BraTS Datenordner nicht gefunden: {brats_data}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Model name: {args.model_name}")

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError("sam2 nicht gefunden. Installieren mit: pip install git+https://github.com/facebookresearch/sam2.git")

    print(f"Lade Modell: {checkpoint.name}")
    sam2_model = build_sam2(args.model_cfg, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("Modell erfolgreich geladen.")

    all_files = load_brats_files(brats_data)
    if len(all_files) == 0:
        raise RuntimeError(f"Keine H5-Dateien in {brats_data} gefunden.")
    print(f"Gefundene H5-Slices: {len(all_files)}")

    if args.only_tumor:
        print("Filtere Slices ohne Tumor...")
        import h5py
        tumor_files = []
        for f in tqdm(all_files, desc="Filtering"):
            try:
                with h5py.File(f, "r") as hf:
                    if hf["mask"][:].sum() > 0:
                        tumor_files.append(f)
            except Exception:
                pass
        print(f"Slices mit Tumor: {len(tumor_files)} / {len(all_files)}")
        all_files = tumor_files

    random.seed(args.seed)
    random.shuffle(all_files)

    if args.max_slices is not None:
        all_files = all_files[:max(1, args.max_slices)]
        print(f"Evaluiere {len(all_files)} Slices.")
    else:
        print(f"Evaluiere alle {len(all_files)} Slices mit Tumor.")

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    saved_examples = {f"{args.model_name}_box_prompt": 0, f"{args.model_name}_point_prompt": 0}

    for h5_path in tqdm(all_files, desc=f"Evaluating {args.model_name}"):
        slice_id = h5_path.stem

        try:
            image_rgb, mask_gt = load_brats_h5(h5_path)
            rows_for_slice = run_prompt_experiments(
                predictor=predictor,
                image=image_rgb,
                mask_gt=mask_gt,
                slice_id=slice_id,
                model_name=args.model_name,
            )
            rows.extend(rows_for_slice)

            box_key = f"{args.model_name}_box_prompt"
            point_key = f"{args.model_name}_point_prompt"

            if rows_for_slice and saved_examples[box_key] < args.save_examples:
                box_row = rows_for_slice[0]
                save_overlay(
                    image=image_rgb,
                    gt_box=[box_row["gt_x1"], box_row["gt_y1"], box_row["gt_x2"], box_row["gt_y2"]],
                    mask_pred=box_row["_mask_pred"],
                    mask_gt=box_row["_mask_gt"],
                    out_path=examples_dir / f"{slice_id}_box.png",
                    title=f"{args.model_name} Box | {slice_id}",
                    prompt_box=box_row["_prompt_box"],
                )
                saved_examples[box_key] += 1

            if rows_for_slice and saved_examples[point_key] < args.save_examples:
                point_row = rows_for_slice[1]
                save_overlay(
                    image=image_rgb,
                    gt_box=[point_row["gt_x1"], point_row["gt_y1"], point_row["gt_x2"], point_row["gt_y2"]],
                    mask_pred=point_row["_mask_pred"],
                    mask_gt=point_row["_mask_gt"],
                    out_path=examples_dir / f"{slice_id}_point.png",
                    title=f"{args.model_name} Point | {slice_id}",
                    prompt_point=point_row["_prompt_point"],
                )
                saved_examples[point_key] += 1

        except Exception as exc:
            rows.append({"slice_id": slice_id, "model": args.model_name, "status": "error", "error": str(exc)})

    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    result_df = pd.DataFrame(clean_rows)
    results_csv = output_dir / f"{args.model_name}_brats_results.csv"
    result_df.to_csv(results_csv, index=False)
    print(f"\nErgebnisse gespeichert: {results_csv}")

    ok_df = result_df[result_df["status"] == "ok"].copy() if "status" in result_df.columns else result_df.copy()

    if len(ok_df) > 0:
        metric_cols = [
            "sam_score", "dice_score", "mask_iou_with_gt_mask",
            "mask_precision_vs_gt_mask", "mask_recall_vs_gt_mask",
            "mask_inside_gt_box", "gt_box_covered_by_mask",
            "pred_box_iou_with_gt_box", "mask_area",
        ]
        available_metrics = [col for col in metric_cols if col in ok_df.columns]
        summary_mean = ok_df.groupby("experiment")[available_metrics].mean()
        summary_median = ok_df.groupby("experiment")[available_metrics].median()
        summary_mean.to_csv(output_dir / f"{args.model_name}_brats_summary_mean.csv")
        summary_median.to_csv(output_dir / f"{args.model_name}_brats_summary_median.csv")
        print("\nMean summary:\n", summary_mean)
        print("\nMedian summary:\n", summary_median)


if __name__ == "__main__":
    main()
    