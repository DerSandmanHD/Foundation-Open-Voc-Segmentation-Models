from __future__ import annotations

import argparse
import json
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
        "--prompt-template",
        default=None,
        help=(
            "Optional template for text prompts, e.g. 'chest x-ray finding: {label}'. "
            "Ignored when --text-prompt is set."
        ),
    )
    parser.add_argument("--run-name", default=None)
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
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument(
        "--example-metric",
        default="pred_box_iou_with_gt_box",
        help="Metric used for best/worst/median example selection.",
    )
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


def resolve_text_prompt(label: str, text_prompt: str | None, prompt_template: str | None) -> str:
    if text_prompt:
        return text_prompt
    if prompt_template:
        return prompt_template.format(label=label)
    return label


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


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
    selected: dict | None,
    prompt_title: str,
    output_path: Path,
    prompt_box: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    if selected is not None:
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
    if selected is not None:
        px1, py1, px2, py2 = selected["box"]
        ax.add_patch(
            plt.Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                fill=False, color="lime", linewidth=2,
            )
        )
        title = f"SAM3: {prompt_title} | Score {selected['score']:.3f}"
    else:
        title = f"SAM3: {prompt_title} | no detection"
    ax.set_title(title)
    ax.axis("off")
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
        metric = "sam3_score" if "sam3_score" in evaluated.columns else "annotation_number"
    evaluated[metric] = pd.to_numeric(evaluated[metric], errors="coerce").fillna(0.0)

    selected: dict[str, pd.DataFrame] = {}
    used_annotation_numbers: set[int] = set()

    def take_unique(frame: pd.DataFrame, count: int) -> pd.DataFrame:
        if count <= 0:
            return frame.head(0)
        rows = []
        for _, row in frame.iterrows():
            annotation_number = int(row["annotation_number"])
            if annotation_number in used_annotation_numbers:
                continue
            used_annotation_numbers.add(annotation_number)
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
    image_index: dict[str, str],
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
            image_name = str(row["image"])
            image_path = image_index.get(image_name)
            if image_path is None:
                continue
            gt_box = [
                float(row["gt_x1"]),
                float(row["gt_y1"]),
                float(row["gt_x2"]),
                float(row["gt_y2"]),
            ]
            prompt = str(row["text_prompt"])
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
            prompt_title = args.prompt_mode if args.prompt_mode == "box" else prompt
            save_overlay(
                image=image,
                gt_box=gt_box,
                selected=selected,
                prompt_title=prompt_title,
                output_path=(
                    output_dir
                    / "examples"
                    / category
                    / f"{int(row['annotation_number']):05d}_{Path(image_name).stem}_sam3.png"
                ),
                prompt_box=gt_box if args.prompt_mode in {"box", "text_box"} else None,
            )
            saved += 1
    return saved


def main() -> None:
    args = parse_args()
    if args.save_examples < 0:
        raise ValueError("--save-examples darf nicht negativ sein.")
    if args.text_prompt and args.prompt_template:
        raise ValueError("--text-prompt und --prompt-template dürfen nicht gemeinsam gesetzt werden.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    run_name = args.run_name or output_dir.name

    for annotation_number, (_, annotation) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc="SAM3 NIH")
    ):
        image_name = str(annotation[image_col])
        label = str(annotation[label_col])
        prompt = resolve_text_prompt(label, args.text_prompt, args.prompt_template)
        base_row = {
            "annotation_number": annotation_number,
            "image": image_name,
            "label": label,
            "dataset": "nih_bbox",
            "model": "sam3",
            "run_name": run_name,
            "experiment": f"sam3_{args.prompt_mode}_prompt",
            "prompt_mode": args.prompt_mode,
            "text_prompt": prompt,
            "prompt_template": args.prompt_template,
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

    saved_examples = 0
    if args.save_examples > 0:
        try:
            saved_examples = save_categorized_examples(
                result_df=result_df,
                image_index=image_index,
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
            "dataset": "NIH ChestX-ray14 BBox_List_2017",
            "evaluation_protocol": (
                "NIH provides bounding boxes, not segmentation masks. Metrics compare "
                "SAM3 masks and boxes against radiologist boxes as proxy metrics."
            ),
            "model": "sam3",
            "run_name": run_name,
            "args": vars(args),
            "paths": {
                "image_root": str(Path(args.image_root).resolve()),
                "bbox_csv": str(Path(args.bbox_csv).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "output_dir": str(output_dir.resolve()),
            },
            "counts": {
                "annotations": int(len(df)),
                "csv_unique_images": int(len(csv_image_names)),
                "matched_unique_images": int(len(matched_image_names)),
                "indexed_images": int(len(image_index)),
                "status": {
                    str(key): int(value)
                    for key, value in result_df["status"].value_counts(dropna=False).items()
                },
                "examples_saved": int(saved_examples),
            },
            "metrics": metric_columns,
            "outputs": {
                "results": "sam3_nih_results.csv",
                "summary": "sam3_nih_summary.csv",
                "summary_by_label": "sam3_nih_summary_by_label.csv",
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
