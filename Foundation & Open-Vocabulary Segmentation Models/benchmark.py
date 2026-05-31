import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import SamModel, SamProcessor


# ============================================================
# Metrics
# ============================================================

def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def mask_to_bbox(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def evaluate_mask_against_gt_box(mask_np, gt_box):
    mask_bool = mask_np > 0.5

    x1, y1, x2, y2 = gt_box
    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(mask_bool.shape[1], round(x2)))
    y2 = int(min(mask_bool.shape[0], round(y2)))

    gt_box_mask = np.zeros_like(mask_bool, dtype=bool)
    gt_box_mask[y1:y2, x1:x2] = True

    mask_area = mask_bool.sum()
    gt_area = gt_box_mask.sum()
    intersection = np.logical_and(mask_bool, gt_box_mask).sum()

    mask_inside_gt = intersection / mask_area if mask_area > 0 else 0.0
    gt_covered_by_mask = intersection / gt_area if gt_area > 0 else 0.0

    pred_box = mask_to_bbox(mask_bool)
    pred_box_iou = box_iou(pred_box, gt_box) if pred_box is not None else 0.0

    return {
        "mask_area": int(mask_area),
        "mask_inside_gt_box": float(mask_inside_gt),
        "gt_box_covered_by_mask": float(gt_covered_by_mask),
        "pred_box_iou_with_gt_box": float(pred_box_iou),
        "pred_box": pred_box,
    }


# ============================================================
# SAM (Segment Anything Model)
# ============================================================

def get_best_sam_mask(outputs, inputs, sam_processor):
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    scores = outputs.iou_scores.detach().cpu()
    best_idx = scores[0, 0].argmax().item()
    best_score = scores[0, 0, best_idx].item()

    mask_np = masks[0][0][best_idx].numpy()
    return mask_np, best_idx, best_score


def run_sam_with_box(image, box, sam_model, sam_processor, device):
    inputs = sam_processor(
        image,
        input_boxes=[[box]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    return get_best_sam_mask(outputs, inputs, sam_processor)


def run_sam_with_point(image, point, sam_model, sam_processor, device):
    inputs = sam_processor(
        image,
        input_points=[[[point[0], point[1]]]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    return get_best_sam_mask(outputs, inputs, sam_processor)


# ============================================================
# Visualization
# ============================================================

def save_overlay(image, gt_box, mask_np, title, out_path, prompt_box=None, prompt_point=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)

    # Ground-Truth (GT) / Radiologist Box
    ax.add_patch(Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--"
    ))

    # Bounding Box Prompt
    if prompt_box is not None:
        ax.add_patch(Rectangle(
            (prompt_box[0], prompt_box[1]),
            prompt_box[2] - prompt_box[0],
            prompt_box[3] - prompt_box[1],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none"
        ))

    # Point Prompt
    if prompt_point is not None:
        ax.scatter([prompt_point[0]], [prompt_point[1]], s=80, marker="x", c="yellow")

    # SAM Generated Mask
    if mask_np is not None:
        mask_show = np.where(mask_np > 0.5, mask_np, np.nan)
        ax.imshow(mask_show, cmap="cool", alpha=0.5)

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


# ============================================================
# Data Loading Utilities
# ============================================================

def build_image_index(image_root):
    image_index = {}

    for root, _, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_index[f] = os.path.join(root, f)

    return image_index


def detect_columns(df):
    cols_lower = {c.strip().lower(): c for c in df.columns}

    def find(*names):
        for name in names:
            key = name.strip().lower()
            if key in cols_lower:
                return cols_lower[key]
        return None

    image_col = find("Image Index", "image_index", "image", "filename")
    label_col = find("Finding Label", "finding_label", "label", "class")

    # NIH BBox CSV formats often feature these specific brackets: Bbox [x, y, w, h]
    x_col = find("Bbox [x", "x", "bbox_x", "x_min")
    y_col = find("y", "bbox_y", "y_min")
    w_col = find("w", "bbox_w", "width")
    h_col = find("h]", "h", "bbox_h", "height")

    needed = [image_col, label_col, x_col, y_col, w_col, h_col]
    if any(c is None for c in needed):
        print("Detected CSV columns:")
        print(list(df.columns))
        raise ValueError("Could not automatically detect CSV columns. Please check column mappings.")

    return image_col, label_col, x_col, y_col, w_col, h_col


# ============================================================
# Main Execution Flow
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory containing the NIH images")
    parser.add_argument("--bbox_csv", type=str, required=True,
                        help="Path to the CSV file containing the expert bounding boxes")
    parser.add_argument("--output_dir", type=str, default="sam_batch_output")
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument("--label", type=str, default=None,
                        help="Optional: filter processing for a single pathology, e.g., Atelectasis")
    parser.add_argument("--save_examples", type=int, default=10)

    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    example_dir = os.path.join(args.output_dir, "examples")
    os.makedirs(example_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Building image index...")
    image_index = build_image_index(args.image_root)
    print("Images found:", len(image_index))

    print("Loading CSV...")
    df = pd.read_csv(args.bbox_csv)
    image_col, label_col, x_col, y_col, w_col, h_col = detect_columns(df)

    # Filter dataset by class if specified
    if args.label is not None:
        df = df[df[label_col].astype(str).str.lower() == args.label.lower()]

    df = df.head(args.max_images).copy()
    print("Boxes to process:", len(df))

    print("Loading SAM...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model.eval()
    print("SAM is ready.")

    rows = []
    saved_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = str(row[image_col])

        if image_name not in image_index:
            rows.append({
                "image": image_name,
                "label": row[label_col],
                "status": "image_not_found"
            })
            continue

        image_path = image_index[image_name]

        try:
            image = Image.open(image_path).convert("RGB")

            # Extract spatial coordinates
            x = float(row[x_col])
            y = float(row[y_col])
            w = float(row[w_col])
            h = float(row[h_col])

            gt_box = [x, y, x + w, y + h]
            center_point = [x + w / 2.0, y + h / 2.0]

            # -----------------------------
            # Experiment 1: Box Prompting
            # -----------------------------
            box_mask, box_best_idx, box_sam_score = run_sam_with_box(
                image, gt_box, sam_model, sam_processor, device
            )
            box_metrics = evaluate_mask_against_gt_box(box_mask, gt_box)

            rows.append({
                "image": image_name,
                "label": row[label_col],
                "experiment": "sam_box_prompt",
                "status": "ok",
                "sam_score": box_sam_score,
                "best_mask_index": box_best_idx,
                "mask_area": box_metrics["mask_area"],
                "mask_inside_gt_box": box_metrics["mask_inside_gt_box"],
                "gt_box_covered_by_mask": box_metrics["gt_box_covered_by_mask"],
                "pred_box_iou_with_gt_box": box_metrics["pred_box_iou_with_gt_box"],
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
            })

            # -----------------------------
            # Experiment 2: Point Prompting
            # -----------------------------
            point_mask, point_best_idx, point_sam_score = run_sam_with_point(
                image, center_point, sam_model, sam_processor, device
            )
            point_metrics = evaluate_mask_against_gt_box(point_mask, gt_box)

            rows.append({
                "image": image_name,
                "label": row[label_col],
                "experiment": "sam_point_prompt",
                "status": "ok",
                "sam_score": point_sam_score,
                "best_mask_index": point_best_idx,
                "mask_area": point_metrics["mask_area"],
                "mask_inside_gt_box": point_metrics["mask_inside_gt_box"],
                "gt_box_covered_by_mask": point_metrics["gt_box_covered_by_mask"],
                "pred_box_iou_with_gt_box": point_metrics["pred_box_iou_with_gt_box"],
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
            })

            # Save qualitative visual overlays
            if saved_count < args.save_examples:
                base = os.path.splitext(image_name)[0]

                save_overlay(
                    image=image,
                    gt_box=gt_box,
                    mask_np=box_mask,
                    title=f"SAM Box-Prompt: {row[label_col]}",
                    out_path=os.path.join(example_dir, f"{base}_box.png"),
                    prompt_box=gt_box
                )

                save_overlay(
                    image=image,
                    gt_box=gt_box,
                    mask_np=point_mask,
                    title=f"SAM Point-Prompt: {row[label_col]}",
                    out_path=os.path.join(example_dir, f"{base}_point.png"),
                    prompt_point=center_point
                )

                saved_count += 1

        except Exception as e:
            rows.append({
                "image": image_name,
                "label": row[label_col],
                "status": "error",
                "error": str(e)
            })

    # Export metrics metadata to CSV
    result_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "sam_batch_results.csv")
    result_df.to_csv(out_csv, index=False)

    print("\nFinished.")
    print("CSV saved to:", out_csv)
    print("Visual examples saved in:", example_dir)

    # Calculate and output summary descriptive statistics
    ok_df = result_df[result_df["status"] == "ok"].copy()

    if len(ok_df) > 0:
        print("\nMean scores grouped by experiment:")
        summary = ok_df.groupby("experiment")[
            [
                "sam_score",
                "mask_inside_gt_box",
                "gt_box_covered_by_mask",
                "pred_box_iou_with_gt_box",
                "mask_area"
            ]
        ].mean()

        print(summary)

        summary_path = os.path.join(args.output_dir, "sam_batch_summary.csv")
        summary.to_csv(summary_path)
        print("Summary stats saved to:", summary_path)


if __name__ == "__main__":
    main()