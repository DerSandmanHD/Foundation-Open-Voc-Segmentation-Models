from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_FILENAMES = {
    "sam3_nih_results.csv": "nih_bbox",
    "sam3_siim_results.csv": "siim_pneumothorax",
}

METRIC_COLUMNS = [
    "sam3_score",
    "sam3_max_score",
    "sam3_mean_score",
    "num_detections",
    "mask_inside_gt_box",
    "gt_box_covered_by_mask",
    "pred_box_iou_with_gt_box",
    "mask_area",
    "dice",
    "mask_iou",
    "pixel_precision",
    "pixel_recall",
    "pred_mask_area",
    "gt_mask_area",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SAM3 benchmark run folders")
    parser.add_argument(
        "--input-root",
        nargs="+",
        required=True,
        help="One or more output roots containing sam3_*_results.csv files.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--by-label-output",
        default=None,
        help="Optional per-label aggregate CSV path. Defaults to '<output>_by_label.csv'.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_result_files(input_roots: list[str]) -> list[Path]:
    result_files: list[Path] = []
    for input_root in input_roots:
        root = Path(input_root).resolve()
        for filename in RESULT_FILENAMES:
            result_files.extend(root.rglob(filename))
    return sorted(set(result_files))


def scalar_from_config(config: dict[str, Any], key: str, fallback: Any = "") -> Any:
    if key in config:
        return config[key]
    args = config.get("args", {})
    if key in args:
        return args[key]
    return fallback


def status_count(df: pd.DataFrame, status: str) -> int:
    if "status" not in df.columns:
        return 0
    return int((df["status"] == status).sum())


def aggregate_one(result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df = pd.read_csv(result_path)
    run_dir = result_path.parent
    config = load_config(run_dir)
    dataset = config.get("dataset") or RESULT_FILENAMES.get(result_path.name, "")
    run_name = (
        config.get("run_name")
        or (df["run_name"].dropna().iloc[0] if "run_name" in df and df["run_name"].notna().any() else run_dir.name)
    )
    prompt_mode = (
        df["prompt_mode"].dropna().iloc[0]
        if "prompt_mode" in df and df["prompt_mode"].notna().any()
        else scalar_from_config(config, "prompt_mode", "")
    )
    text_prompt = (
        df["text_prompt"].dropna().iloc[0]
        if "text_prompt" in df and df["text_prompt"].notna().any()
        else scalar_from_config(config, "text_prompt", "")
    )
    prompt_template = (
        df["prompt_template"].dropna().iloc[0]
        if "prompt_template" in df and df["prompt_template"].notna().any()
        else scalar_from_config(config, "prompt_template", "")
    )
    threshold = (
        df["confidence_threshold"].dropna().iloc[0]
        if "confidence_threshold" in df and df["confidence_threshold"].notna().any()
        else scalar_from_config(config, "threshold", "")
    )

    evaluated = df[df["status"].isin(["ok", "no_detection"])].copy()
    aggregate: dict[str, Any] = {
        "dataset": dataset,
        "run_name": run_name,
        "prompt_mode": prompt_mode,
        "text_prompt": text_prompt,
        "prompt_template": prompt_template,
        "confidence_threshold": threshold,
        "result_path": str(result_path),
        "run_dir": str(run_dir),
        "n_total": int(len(df)),
        "n_evaluated": int(len(evaluated)),
        "n_ok": status_count(df, "ok"),
        "n_no_detection": status_count(df, "no_detection"),
        "n_error": status_count(df, "error"),
        "n_image_not_found": status_count(df, "image_not_found"),
    }
    metric_columns = [column for column in METRIC_COLUMNS if column in evaluated.columns]
    for column in metric_columns:
        values = pd.to_numeric(evaluated[column], errors="coerce")
        aggregate[f"{column}_mean"] = float(values.mean()) if values.notna().any() else pd.NA
        aggregate[f"{column}_median"] = float(values.median()) if values.notna().any() else pd.NA
        aggregate[f"{column}_std"] = float(values.std()) if values.notna().any() else pd.NA

    by_label_rows: list[dict[str, Any]] = []
    if "label" in evaluated.columns:
        for label, label_df in evaluated.groupby("label", dropna=False):
            label_row: dict[str, Any] = {
                "dataset": dataset,
                "run_name": run_name,
                "prompt_mode": prompt_mode,
                "text_prompt": text_prompt,
                "prompt_template": prompt_template,
                "confidence_threshold": threshold,
                "label": label,
                "n_evaluated": int(len(label_df)),
                "n_ok": status_count(label_df, "ok"),
                "n_no_detection": status_count(label_df, "no_detection"),
            }
            for column in metric_columns:
                values = pd.to_numeric(label_df[column], errors="coerce")
                label_row[f"{column}_mean"] = (
                    float(values.mean()) if values.notna().any() else pd.NA
                )
                label_row[f"{column}_median"] = (
                    float(values.median()) if values.notna().any() else pd.NA
                )
            by_label_rows.append(label_row)

    return aggregate, by_label_rows


def main() -> None:
    args = parse_args()
    result_files = find_result_files(args.input_root)
    if not result_files:
        raise RuntimeError(f"Keine Result-Dateien gefunden unter: {args.input_root}")

    aggregate_rows: list[dict[str, Any]] = []
    by_label_rows: list[dict[str, Any]] = []
    for result_path in result_files:
        aggregate, label_rows = aggregate_one(result_path)
        aggregate_rows.append(aggregate)
        by_label_rows.extend(label_rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aggregate_rows).sort_values(["dataset", "run_name"]).to_csv(
        output_csv,
        index=False,
    )

    by_label_output = (
        Path(args.by_label_output)
        if args.by_label_output
        else output_csv.with_name(f"{output_csv.stem}_by_label.csv")
    )
    if by_label_rows:
        pd.DataFrame(by_label_rows).sort_values(["dataset", "run_name", "label"]).to_csv(
            by_label_output,
            index=False,
        )

    print(f"Aggregierte Runs: {len(aggregate_rows)}")
    print(f"Run-Aggregation: {output_csv}")
    if by_label_rows:
        print(f"Label-Aggregation: {by_label_output}")


if __name__ == "__main__":
    main()
