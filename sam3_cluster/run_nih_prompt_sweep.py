from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple NIH SAM3 prompt benchmarks")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--bbox-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--sweep-csv",
        default=str(Path(__file__).with_name("nih_prompt_sweep.csv")),
    )
    parser.add_argument("--output-root", default="sam3_outputs/nih_prompt_sweep")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--default-threshold", type=float, default=0.0)
    parser.add_argument("--max-annotations", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=10)
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument("--example-metric", default="pred_box_iou_with_gt_box")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_sweep_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError(f"Prompt-Sweep ist leer: {path}")
    for index, row in enumerate(rows, start=1):
        if not row.get("run_name"):
            raise ValueError(f"run_name fehlt in Sweep-Zeile {index}")
        if row.get("prompt_mode") not in {"box", "text", "text_box"}:
            raise ValueError(
                f"Ungültiger prompt_mode in Sweep-Zeile {index}: {row.get('prompt_mode')}"
            )
        if row.get("text_prompt") and row.get("prompt_template"):
            raise ValueError(
                f"Setze entweder text_prompt oder prompt_template, nicht beide: {row['run_name']}"
            )
    return rows


def build_run_command(
    args: argparse.Namespace,
    row: dict[str, str],
    output_dir: Path,
) -> list[str]:
    threshold = row.get("threshold") or str(args.default_threshold)
    command = [
        sys.executable,
        "-m",
        "sam3_cluster.nih_text_benchmark",
        "--image-root",
        args.image_root,
        "--bbox-csv",
        args.bbox_csv,
        "--checkpoint",
        args.checkpoint,
        "--output-dir",
        str(output_dir),
        "--run-name",
        row["run_name"],
        "--prompt-mode",
        row["prompt_mode"],
        "--threshold",
        threshold,
        "--precision",
        args.precision,
        "--save-examples",
        str(args.save_examples),
        "--example-seed",
        str(args.example_seed),
        "--example-metric",
        args.example_metric,
    ]
    if args.max_annotations is not None:
        command.extend(["--max-annotations", str(args.max_annotations)])
    if row.get("text_prompt"):
        command.extend(["--text-prompt", row["text_prompt"]])
    if row.get("prompt_template"):
        command.extend(["--prompt-template", row["prompt_template"]])
    if row.get("label"):
        command.extend(["--label", row["label"]])
    return command


def main() -> None:
    args = parse_args()
    sweep_csv = Path(args.sweep_csv).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_sweep_rows(sweep_csv)

    manifest: list[dict] = []
    for row in rows:
        output_dir = output_root / row["run_name"]
        command = build_run_command(args, row, output_dir)
        print(f"\n=== NIH Sweep: {row['run_name']} ===")
        print(" ".join(command))
        manifest_row = {
            "run_name": row["run_name"],
            "prompt_mode": row["prompt_mode"],
            "output_dir": str(output_dir),
            "command": command,
            "status": "dry_run" if args.dry_run else "pending",
        }
        if args.dry_run:
            manifest.append(manifest_row)
            continue
        try:
            subprocess.run(command, check=True)
            manifest_row["status"] = "ok"
        except subprocess.CalledProcessError as exc:
            manifest_row["status"] = "error"
            manifest_row["returncode"] = exc.returncode
            manifest.append(manifest_row)
            if not args.continue_on_error:
                raise
            continue
        manifest.append(manifest_row)

    manifest_path = output_root / "sweep_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    if not args.no_aggregate and not args.dry_run:
        aggregate_csv = output_root / "sam3_nih_sweep_aggregate.csv"
        aggregate_command = [
            sys.executable,
            "-m",
            "sam3_cluster.aggregate_runs",
            "--input-root",
            str(output_root),
            "--output-csv",
            str(aggregate_csv),
        ]
        print("\n=== Aggregation ===")
        print(" ".join(aggregate_command))
        subprocess.run(aggregate_command, check=True)

    print(f"\nSweep-Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
