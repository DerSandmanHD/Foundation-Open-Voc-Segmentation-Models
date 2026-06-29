from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sam3_cluster.common import (
    cuda_inference_context,
    load_image_model,
    tensor_to_numpy,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 text benchmark on SIIM masks")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--text-prompt", default="pneumothorax")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", required=True, help="Local SAM3 sam3.pt")
    parser.add_argument(
        "--precision", choices=["bf16", "fp16", "fp32"], default="fp16"
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=10)
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="sam3_outputs/siim_pneumothorax")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def count_supported_files(directory: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.rglob("*")
    )


def discover_image_and_mask_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if bool(args.image_dir) != bool(args.mask_dir):
        raise ValueError("--image-dir und --mask-dir müssen gemeinsam gesetzt werden.")
    if args.image_dir and args.mask_dir:
        image_dir = Path(args.image_dir).resolve()
        mask_dir = Path(args.mask_dir).resolve()
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Bild- oder Maskenordner fehlt: {image_dir}, {mask_dir}"
            )
        return image_dir, mask_dir

    data_root = Path(args.data_root).resolve()
    candidates: list[tuple[int, Path, Path]] = []
    for image_dir in data_root.rglob("dicom"):
        mask_dir = image_dir.parent / "mask"
        if image_dir.is_dir() and mask_dir.is_dir():
            candidates.append(
                (
                    min(
                        count_supported_files(image_dir),
                        count_supported_files(mask_dir),
                    ),
                    image_dir,
                    mask_dir,
                )
            )
    if not candidates:
        raise RuntimeError(
            f"Unter {data_root} wurde kein Ordnerpaar 'dicom'/'mask' gefunden. "
            "Setze --image-dir und --mask-dir explizit."
        )
    _, image_dir, mask_dir = max(candidates, key=lambda item: item[0])
    return image_dir, mask_dir


def build_file_index(directory: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def segmentation_metrics(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)
    intersection = int(np.logical_and(prediction, ground_truth).sum())
    pred_area = int(prediction.sum())
    gt_area = int(ground_truth.sum())
    union = pred_area + gt_area - intersection
    return {
        "dice": (
            2.0 * intersection / (pred_area + gt_area)
            if pred_area + gt_area
            else 1.0
        ),
        "mask_iou": intersection / union if union else 1.0,
        "pixel_precision": intersection / pred_area if pred_area else 0.0,
        "pixel_recall": intersection / gt_area if gt_area else 0.0,
        "pred_mask_area": float(pred_area),
        "gt_mask_area": float(gt_area),
    }


def combine_detections(output: dict, image_shape: tuple[int, int]) -> dict:
    scores = tensor_to_numpy(output["scores"]).reshape(-1)
    if scores.size == 0:
        return {
            "mask": np.zeros(image_shape, dtype=bool),
            "num_detections": 0,
            "max_score": 0.0,
            "mean_score": 0.0,
        }
    masks = tensor_to_numpy(output["masks"]).astype(bool)
    masks = masks.reshape((-1,) + image_shape)
    return {
        "mask": np.any(masks, axis=0),
        "num_detections": int(scores.size),
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
    }


def save_overlay(
    image: Image.Image,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Chest X-ray")
    axes[1].imshow(image)
    axes[1].imshow(
        np.ma.masked_where(~ground_truth, ground_truth), cmap="Reds", alpha=0.55
    )
    axes[1].set_title("Ground truth")
    axes[2].imshow(image)
    axes[2].imshow(
        np.ma.masked_where(~prediction, prediction), cmap="Blues", alpha=0.55
    )
    axes[2].contour(ground_truth, levels=[0.5], colors=["red"], linewidths=1.2)
    axes[2].set_title(
        f"SAM3 prediction\nDice {metrics['dice']:.3f} | IoU {metrics['mask_iou']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images muss größer als 0 sein.")
    if args.save_examples < 0:
        raise ValueError("--save-examples darf nicht negativ sein.")

    output_dir = Path(args.output_dir)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sam3_siim_results.csv"

    image_dir, mask_dir = discover_image_and_mask_dirs(args)
    image_index = build_file_index(image_dir)
    mask_index = build_file_index(mask_dir)
    paired_names = sorted(image_index.keys() & mask_index.keys())
    if not paired_names:
        raise RuntimeError("Keine Bild-/Maskenpaare mit identischen Dateinamen gefunden.")

    positive_names: list[str] = []
    for name in tqdm(paired_names, desc="SIIM-Masken prüfen"):
        mask = np.asarray(Image.open(mask_index[name]).convert("L"), dtype=np.uint8)
        if np.any(mask):
            positive_names.append(name)
            if args.max_images is not None and len(positive_names) >= args.max_images:
                break
    if not positive_names:
        raise RuntimeError("Keine nicht-leeren SIIM-Masken gefunden.")

    example_count = min(args.save_examples, len(positive_names))
    example_names = set(
        random.Random(args.example_seed).sample(positive_names, example_count)
    )
    print(f"SIIM-Datenwurzel: {Path(args.data_root).resolve()}")
    print(f"Bildordner: {image_dir}")
    print(f"Maskenordner: {mask_dir}")
    print(f"Bild-/Maskenpaare: {len(paired_names)}")
    print(f"Positive Fälle für Benchmark: {len(positive_names)}")
    print(f"Zu speichernde Visualisierungen: {example_count}")

    _, processor, _ = load_image_model(args.checkpoint, args.threshold)
    rows: list[dict] = []
    for image_number, name in enumerate(tqdm(positive_names, desc="SAM3 SIIM")):
        base_row = {
            "image_number": image_number,
            "image": name,
            "experiment": "sam3_text_prompt",
            "text_prompt": args.text_prompt,
            "confidence_threshold": args.threshold,
            "precision": args.precision,
        }
        try:
            image = Image.open(image_index[name]).convert("RGB")
            ground_truth = np.asarray(Image.open(mask_index[name]).convert("L")) > 0
            with cuda_inference_context(args.precision):
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=args.text_prompt)
            selected = combine_detections(output, ground_truth.shape)
            metrics = segmentation_metrics(selected["mask"], ground_truth)
            rows.append(
                {
                    **base_row,
                    "status": "ok" if selected["num_detections"] else "no_detection",
                    "num_detections": selected["num_detections"],
                    "sam3_max_score": selected["max_score"],
                    "sam3_mean_score": selected["mean_score"],
                    **metrics,
                }
            )
            if name in example_names:
                save_overlay(
                    image,
                    ground_truth,
                    selected["mask"],
                    metrics,
                    examples_dir / f"{image_number:05d}_{Path(name).stem}_sam3.png",
                )
        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append(
                {
                    **base_row,
                    "status": "error",
                    "error": str(exc),
                    "num_detections": np.nan,
                    "sam3_max_score": np.nan,
                    "sam3_mean_score": np.nan,
                    "dice": np.nan,
                    "mask_iou": np.nan,
                    "pixel_precision": np.nan,
                    "pixel_recall": np.nan,
                    "pred_mask_area": np.nan,
                    "gt_mask_area": np.nan,
                }
            )

        if rows and len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(results_path, index=False)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(results_path, index=False)
    evaluated = result_df[result_df["status"].isin(["ok", "no_detection"])]
    metric_columns = [
        "sam3_max_score",
        "num_detections",
        "dice",
        "mask_iou",
        "pixel_precision",
        "pixel_recall",
        "pred_mask_area",
        "gt_mask_area",
    ]
    summary = evaluated[metric_columns].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).T
    summary.to_csv(output_dir / "sam3_siim_summary.csv")
    print(result_df["status"].value_counts(dropna=False))
    print(summary)
    print(f"Ergebnisse: {output_dir}")


if __name__ == "__main__":
    main()
