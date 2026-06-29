from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sam3_cluster.common import (
    cuda_inference_context,
    evaluate_mask_against_box,
    load_image_model,
    select_highest_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 benchmark on NIH CXR boxes")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bbox-csv", required=True)
    parser.add_argument(
        "--label",
        default=None,
        help="Optional pathology filter. By default all BBox annotations are used.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["box", "text", "text_box"],
        default="box",
        help=(
            "box: NIH GT box as SAM3 geometric prompt, like benchmark.py. "
            "text: SAM3 open-vocabulary text prompt. "
            "text_box: text prompt plus NIH GT box prompt."
        ),
    )
    parser.add_argument("--text-prompt", default=None)
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=None,
        help="Optional limit for test runs. Omit it to process all annotations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "SAM3 confidence threshold. Default 0.0 mirrors benchmark.py by "
            "keeping candidates and selecting the highest score."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Local SAM3 sam3.pt")
    parser.add_argument(
        "--precision", choices=["bf16", "fp16", "fp32"], default="fp16"
    )
    parser.add_argument("--output-dir", default="sam3_outputs/nih_text")
    parser.add_argument("--save-examples", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def detect_columns(df: pd.DataFrame) -> tuple[str, str, str, str, str, str]:
    normalized = {column.strip().lower(): column for column in df.columns}

    def find(*names: str) -> str:
        for name in names:
            if name.lower() in normalized:
                return normalized[name.lower()]
        raise ValueError(f"Keine passende Spalte für {names}; vorhanden: {list(df.columns)}")

    return (
        find("Image Index", "image", "filename"),
        find("Finding Label", "label", "class"),
        find("Bbox [x", "x", "bbox_x", "x_min"),
        find("y", "bbox_y", "y_min"),
        find("w", "bbox_w", "width"),
        find("h]", "h", "bbox_h", "height"),
    )


def build_image_index(image_root: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for root, _, files in os.walk(image_root):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                index[filename] = os.path.join(root, filename)
    return index


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


def save_overlay(
    image: Image.Image,
    gt_box: list[float],
    selected: dict,
    prompt_title: str,
    output_path: Path,
    prompt_box: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    ax.imshow(selected["mask"], cmap="jet", alpha=0.4)
    gx1, gy1, gx2, gy2 = gt_box
    ax.add_patch(
        plt.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            fill=False, color="red", linestyle="--", linewidth=2,
        )
    )
    if prompt_box is not None:
        bx1, by1, bx2, by2 = prompt_box
        ax.add_patch(
            plt.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                fill=False, color="yellow", linewidth=2,
            )
        )
    px1, py1, px2, py2 = selected["box"]
    ax.add_patch(
        plt.Rectangle(
            (px1, py1), px2 - px1, py2 - py1,
            fill=False, color="lime", linewidth=2,
        )
    )
    ax.set_title(f"SAM3: {prompt_title} | Score {selected['score']:.3f}")
    ax.axis("off")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    if args.save_examples > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.bbox_csv)
    image_col, label_col, x_col, y_col, w_col, h_col = detect_columns(df)
    if args.label:
        df = df[df[label_col].astype(str).str.casefold() == args.label.casefold()]
    if args.max_annotations is not None:
        if args.max_annotations <= 0:
            raise ValueError("--max-annotations muss größer als 0 sein.")
        df = df.head(args.max_annotations)
    df = df.copy()
    if df.empty:
        raise RuntimeError("Nach dem Filter sind keine Annotationen übrig.")

    image_index = build_image_index(args.image_root)
    csv_image_names = set(df[image_col].astype(str))
    matched_image_names = csv_image_names.intersection(image_index)
    print(f"Bildwurzel: {Path(args.image_root).resolve()}")
    print(f"Gefundene Bilddateien: {len(image_index)}")
    print(f"Passende CSV-Bilder: {len(matched_image_names)} / {len(csv_image_names)}")
    if not matched_image_names:
        indexed_examples = sorted(image_index)[:5]
        csv_examples = sorted(csv_image_names)[:5]
        raise RuntimeError(
            "Kein einziger Dateiname aus der BBox-CSV wurde unter --image-root "
            f"gefunden. Beispiele im Bildordner: {indexed_examples or 'keine'}; "
            f"Beispiele aus der CSV: {csv_examples}. Prüfe, ob die NIH-Archive "
            "wirklich bis zu den PNG-Dateien entpackt wurden und ob "
            "SAM3_NIH_ROOT auf deren gemeinsamen Oberordner zeigt."
        )
    _, processor, _ = load_image_model(args.checkpoint, args.threshold)
    rows: list[dict] = []
    saved_examples = 0

    for annotation_number, (_, annotation) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="SAM3 NIH")
    ):
        image_name = str(annotation[image_col])
        label = str(annotation[label_col])
        prompt = args.text_prompt or label
        base_row = {
            "annotation_number": annotation_number,
            "image": image_name,
            "label": label,
            "experiment": f"sam3_{args.prompt_mode}_prompt",
            "prompt_mode": args.prompt_mode,
            "text_prompt": prompt,
            "confidence_threshold": args.threshold,
            "precision": args.precision,
        }
        image_path = image_index.get(image_name)
        if image_path is None:
            rows.append({**base_row, "status": "image_not_found", "num_detections": 0})
            continue

        x = float(annotation[x_col])
        y = float(annotation[y_col])
        width = float(annotation[w_col])
        height = float(annotation[h_col])
        gt_box = [x, y, x + width, y + height]
        base_row.update(
            {
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
            }
        )

        try:
            image = Image.open(image_path).convert("RGB")
            output = run_sam3_prompt(
                processor=processor,
                image=image,
                gt_box=gt_box,
                prompt_mode=args.prompt_mode,
                text_prompt=prompt,
                precision=args.precision,
            )
            selected = select_highest_score(output)

            if selected is None:
                rows.append(
                    {
                        **base_row,
                        "status": "no_detection",
                        "num_detections": 0,
                        "sam3_score": 0.0,
                        "mask_area": 0.0,
                        "mask_inside_gt_box": 0.0,
                        "gt_box_covered_by_mask": 0.0,
                        "pred_box_iou_with_gt_box": 0.0,
                    }
                )
                continue

            metrics = evaluate_mask_against_box(selected["mask"], gt_box)
            rows.append(
                {
                    **base_row,
                    "status": "ok",
                    "num_detections": selected["num_detections"],
                    "sam3_score": selected["score"],
                    **metrics,
                    "pred_x1": selected["box"][0],
                    "pred_y1": selected["box"][1],
                    "pred_x2": selected["box"][2],
                    "pred_y2": selected["box"][3],
                }
            )
            if saved_examples < args.save_examples:
                save_overlay(
                    image,
                    gt_box,
                    selected,
                    args.prompt_mode if args.prompt_mode == "box" else prompt,
                    examples_dir / f"{annotation_number:05d}_{Path(image_name).stem}_sam3.png",
                    prompt_box=gt_box if args.prompt_mode in {"box", "text_box"} else None,
                )
                saved_examples += 1
        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append({**base_row, "status": "error", "error": str(exc)})

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "sam3_nih_results.csv", index=False)

    evaluated = result_df[result_df["status"].isin(["ok", "no_detection"])].copy()
    metric_columns = [
        "sam3_score",
        "mask_inside_gt_box",
        "gt_box_covered_by_mask",
        "pred_box_iou_with_gt_box",
        "mask_area",
    ]
    if evaluated.empty:
        summary = pd.DataFrame(index=metric_columns, columns=["mean", "median"])
    else:
        summary = evaluated[metric_columns].mean().to_frame(name="mean")
        summary["median"] = evaluated[metric_columns].median()
    summary.to_csv(output_dir / "sam3_nih_summary.csv")

    if evaluated.empty:
        summary_by_label = pd.DataFrame()
    else:
        summary_by_label = evaluated.groupby("label")[metric_columns].agg(
            ["count", "mean", "median"]
        )
    summary_by_label.to_csv(output_dir / "sam3_nih_summary_by_label.csv")

    print(result_df["status"].value_counts(dropna=False))
    print(summary)
    print(f"Ergebnisse: {output_dir}")


if __name__ == "__main__":
    main()
