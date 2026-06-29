from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="SAM3 benchmark on SIIM masks")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=["text", "box", "text_box"],
        default="text",
        help=(
            "text: SAM3 open-vocabulary text prompt. "
            "box: GT mask bounding box as SAM3 geometric prompt. "
            "text_box: text prompt plus GT mask bounding box."
        ),
    )
    parser.add_argument("--text-prompt", default="pneumothorax")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--checkpoint", required=True, help="Local SAM3 sam3.pt")
    parser.add_argument(
        "--precision", choices=["bf16", "fp16", "fp32"], default="fp16"
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=10)
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument("--example-metric", default="mask_iou")
    parser.add_argument("--output-dir", default="sam3_outputs/siim_pneumothorax")
    parser.add_argument("--run-name", default=None)
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


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


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


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def xyxy_to_normalized_cxcywh(box: list[float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = box
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0
    return [
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    ]


def run_sam3_prompt(
    processor,
    image: Image.Image,
    gt_box: list[float],
    prompt_mode: str,
    text_prompt: str,
    precision: str,
) -> dict:
    with cuda_inference_context(precision):
        state = processor.set_image(image)
        if prompt_mode in {"text", "text_box"}:
            state = processor.set_text_prompt(state=state, prompt=text_prompt)
        if prompt_mode in {"box", "text_box"}:
            prompt_box = xyxy_to_normalized_cxcywh(gt_box, image.size)
            state = processor.add_geometric_prompt(
                box=prompt_box,
                label=True,
                state=state,
            )
    return state


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
    prompt_box: list[float] | None = None,
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
    if prompt_box is not None:
        x1, y1, x2, y2 = prompt_box
        axes[0].add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                color="yellow",
                linewidth=2,
            )
        )
        axes[2].add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                color="yellow",
                linewidth=2,
            )
        )
    axes[2].set_title(
        f"SAM3 prediction\nDice {metrics['dice']:.3f} | IoU {metrics['mask_iou']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def allocate_example_budget(total: int, categories: list[str]) -> dict[str, int]:
    if total <= 0:
        return {category: 0 for category in categories}
    base = total // len(categories)
    remainder = total % len(categories)
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }


def select_example_rows(
    result_df: pd.DataFrame,
    total_examples: int,
    metric: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    categories = ["random", "best", "worst", "median"]
    budget = allocate_example_budget(total_examples, categories)
    evaluated = result_df[result_df["status"].isin(["ok", "no_detection"])].copy()
    if evaluated.empty:
        return {category: evaluated.head(0) for category in categories}
    if metric not in evaluated.columns:
        metric = "mask_iou" if "mask_iou" in evaluated.columns else "image_number"
    evaluated[metric] = pd.to_numeric(evaluated[metric], errors="coerce").fillna(0.0)

    selected: dict[str, pd.DataFrame] = {}
    used_image_numbers: set[int] = set()

    def take_unique(frame: pd.DataFrame, count: int) -> pd.DataFrame:
        if count <= 0:
            return frame.head(0)
        rows = []
        for _, row in frame.iterrows():
            image_number = int(row["image_number"])
            if image_number in used_image_numbers:
                continue
            used_image_numbers.add(image_number)
            rows.append(row)
            if len(rows) >= count:
                break
        return pd.DataFrame(rows, columns=frame.columns)

    selected["random"] = take_unique(
        evaluated.sample(frac=1.0, random_state=seed),
        budget["random"],
    )
    selected["best"] = take_unique(
        evaluated.sort_values(metric, ascending=False),
        budget["best"],
    )
    selected["worst"] = take_unique(
        evaluated.sort_values(metric, ascending=True),
        budget["worst"],
    )
    median_value = float(evaluated[metric].median())
    selected["median"] = take_unique(
        evaluated.assign(_distance=(evaluated[metric] - median_value).abs())
        .sort_values("_distance")
        .drop(columns=["_distance"]),
        budget["median"],
    )
    return selected


def save_categorized_examples(
    result_df: pd.DataFrame,
    image_index: dict[str, Path],
    mask_index: dict[str, Path],
    processor,
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    saved = 0
    selected_by_category = select_example_rows(
        result_df=result_df,
        total_examples=args.save_examples,
        metric=args.example_metric,
        seed=args.example_seed,
    )
    for category, category_df in selected_by_category.items():
        for _, row in category_df.iterrows():
            name = str(row["image"])
            if name not in image_index or name not in mask_index:
                continue
            image = Image.open(image_index[name]).convert("RGB")
            ground_truth = np.asarray(Image.open(mask_index[name]).convert("L")) > 0
            gt_box = mask_to_box(ground_truth)
            if gt_box is None:
                continue
            output = run_sam3_prompt(
                processor=processor,
                image=image,
                gt_box=gt_box,
                prompt_mode=args.prompt_mode,
                text_prompt=args.text_prompt,
                precision=args.precision,
            )
            selected = combine_detections(output, ground_truth.shape)
            metrics = segmentation_metrics(selected["mask"], ground_truth)
            save_overlay(
                image=image,
                ground_truth=ground_truth,
                prediction=selected["mask"],
                metrics=metrics,
                output_path=(
                    output_dir
                    / "examples"
                    / category
                    / f"{int(row['image_number']):05d}_{Path(name).stem}_sam3.png"
                ),
                prompt_box=gt_box if args.prompt_mode in {"box", "text_box"} else None,
            )
            saved += 1
    return saved


def main() -> None:
    args = parse_args()
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images muss größer als 0 sein.")
    if args.save_examples < 0:
        raise ValueError("--save-examples darf nicht negativ sein.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "sam3_siim_results.csv"
    run_name = args.run_name or output_dir.name

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

    print(f"SIIM-Datenwurzel: {Path(args.data_root).resolve()}")
    print(f"Bildordner: {image_dir}")
    print(f"Maskenordner: {mask_dir}")
    print(f"Bild-/Maskenpaare: {len(paired_names)}")
    print(f"Positive Fälle für Benchmark: {len(positive_names)}")
    print(f"Zu speichernde Visualisierungen: {min(args.save_examples, len(positive_names))}")

    _, processor, _ = load_image_model(args.checkpoint, args.threshold)
    rows: list[dict] = []
    for image_number, name in enumerate(tqdm(positive_names, desc="SAM3 SIIM")):
        base_row = {
            "image_number": image_number,
            "image": name,
            "dataset": "siim_pneumothorax",
            "model": "sam3",
            "run_name": run_name,
            "experiment": f"sam3_{args.prompt_mode}_prompt",
            "prompt_mode": args.prompt_mode,
            "text_prompt": args.text_prompt,
            "confidence_threshold": args.threshold,
            "precision": args.precision,
        }
        try:
            image = Image.open(image_index[name]).convert("RGB")
            ground_truth = np.asarray(Image.open(mask_index[name]).convert("L")) > 0
            gt_box = mask_to_box(ground_truth)
            if gt_box is None:
                rows.append(
                    {
                        **base_row,
                        "status": "empty_mask",
                        "num_detections": 0,
                        "sam3_max_score": 0.0,
                        "sam3_mean_score": 0.0,
                        "dice": 0.0,
                        "mask_iou": 0.0,
                        "pixel_precision": 0.0,
                        "pixel_recall": 0.0,
                        "pred_mask_area": 0.0,
                        "gt_mask_area": 0.0,
                    }
                )
                continue
            output = run_sam3_prompt(
                processor=processor,
                image=image,
                gt_box=gt_box,
                prompt_mode=args.prompt_mode,
                text_prompt=args.text_prompt,
                precision=args.precision,
            )
            selected = combine_detections(output, ground_truth.shape)
            metrics = segmentation_metrics(selected["mask"], ground_truth)
            rows.append(
                {
                    **base_row,
                    "status": "ok" if selected["num_detections"] else "no_detection",
                    "num_detections": selected["num_detections"],
                    "sam3_max_score": selected["max_score"],
                    "sam3_mean_score": selected["mean_score"],
                    "gt_x1": gt_box[0],
                    "gt_y1": gt_box[1],
                    "gt_x2": gt_box[2],
                    "gt_y2": gt_box[3],
                    **metrics,
                }
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
    saved_examples = 0
    if args.save_examples > 0:
        try:
            saved_examples = save_categorized_examples(
                result_df=result_df,
                image_index=image_index,
                mask_index=mask_index,
                processor=processor,
                args=args,
                output_dir=output_dir,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            print(f"Warnung: Beispielbilder konnten nicht vollständig gespeichert werden: {exc}")

    write_json(
        output_dir / "config.json",
        {
            "dataset": "SIIM Pneumothorax",
            "evaluation_protocol": (
                "SIIM provides segmentation masks. Metrics compare the union of SAM3 "
                "detections against non-empty pneumothorax masks. In box/text_box "
                "mode, the bounding box is derived from the ground-truth mask."
            ),
            "model": "sam3",
            "run_name": run_name,
            "prompt_mode": args.prompt_mode,
            "args": vars(args),
            "paths": {
                "data_root": str(Path(args.data_root).resolve()),
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "output_dir": str(output_dir.resolve()),
            },
            "counts": {
                "paired_images": int(len(paired_names)),
                "positive_images": int(len(positive_names)),
                "status": {
                    str(key): int(value)
                    for key, value in result_df["status"].value_counts(dropna=False).items()
                },
                "examples_saved": int(saved_examples),
            },
            "metrics": metric_columns,
            "outputs": {
                "results": "sam3_siim_results.csv",
                "summary": "sam3_siim_summary.csv",
                "examples": "examples/",
            },
        },
    )
    print(result_df["status"].value_counts(dropna=False))
    print(summary)
    if args.save_examples > 0:
        print(f"Beispielbilder gespeichert: {saved_examples}")
    print(f"Ergebnisse: {output_dir}")


if __name__ == "__main__":
    main()
