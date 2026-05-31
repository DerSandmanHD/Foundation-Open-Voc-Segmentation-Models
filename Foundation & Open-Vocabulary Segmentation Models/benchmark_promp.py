import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from benchmark import (
    build_image_index,
    detect_columns,
    run_sam_with_box,
    evaluate_mask_against_gt_box,
    box_iou,
)

from transformers import SamModel, SamProcessor


def save_text_overlay(image, gt_box, text_box, mask_np, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)

    # Ground-Truth (GT) Bounding Box
    ax.add_patch(Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label="GT box"
    ))

    # GroundingDINO Predicted Text Box
    if text_box is not None:
        ax.add_patch(Rectangle(
            (text_box[0], text_box[1]),
            text_box[2] - text_box[0],
            text_box[3] - text_box[1],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="-",
            label="Text box"
        ))

    # SAM Generated Mask
    if mask_np is not None:
        mask_show = np.where(mask_np > 0.5, mask_np, np.nan)
        ax.imshow(mask_show, cmap="cool", alpha=0.5)

    ax.legend(loc="lower right")
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def run_groundingdino(image, text_prompt, gd_model, gd_processor, device, img_w, img_h):
    text = text_prompt.strip().lower()
    if not text.endswith("."):
        text += "."

    inputs = gd_processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    # Handle different versions of post-processing function arguments
    try:
        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(img_h, img_w)]
        )[0]
    except TypeError:
        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.20,
            text_threshold=0.20,
            target_sizes=[(img_h, img_w)]
        )[0]

    boxes = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    labels = results["labels"]

    if len(boxes) == 0:
        return None, None, None

    # Return the prediction with the highest confidence score
    best_id = int(np.argmax(scores))
    return boxes[best_id].tolist(), float(scores[best_id]), labels[best_id]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--bbox_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="text_prompt_output")
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--label", type=str, default="Atelectasis")
    parser.add_argument("--text_prompt", type=str, default=None)
    parser.add_argument("--save_examples", type=int, default=10)

    args = parser.parse_args()

    # Create output directories
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

    # Filter by specific label if requested
    if args.label is not None:
        df = df[df[label_col].astype(str).str.lower() == args.label.lower()]

    df = df.head(args.max_images).copy()
    print("Boxes to process:", len(df))

    print("Loading SAM...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model.eval()

    print("Loading GroundingDINO...")
    gd_model_id = "IDEA-Research/grounding-dino-tiny"
    gd_processor = AutoProcessor.from_pretrained(gd_model_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(device)
    gd_model.eval()

    rows = []
    saved_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_name = str(row[image_col])
        label = str(row[label_col])

        if image_name not in image_index:
            rows.append({
                "image": image_name,
                "label": label,
                "experiment": "text_groundingdino_sam",
                "status": "image_not_found"
            })
            continue

        try:
            image = Image.open(image_index[image_name]).convert("RGB")
            img_w, img_h = image.size

            # Parse Ground-Truth coordinates
            x = float(row[x_col])
            y = float(row[y_col])
            w = float(row[w_col])
            h = float(row[h_col])
            gt_box = [x, y, x + w, y + h]

            # Use custom prompt if provided, otherwise default to the dataset label
            text_prompt = args.text_prompt if args.text_prompt else label

            # Step 1: Run text-to-bbox grounding using GroundingDINO
            text_box, gd_score, gd_label = run_groundingdino(
                image=image,
                text_prompt=text_prompt,
                gd_model=gd_model,
                gd_processor=gd_processor,
                device=device,
                img_w=img_w,
                img_h=img_h
            )

            if text_box is None:
                rows.append({
                    "image": image_name,
                    "label": label,
                    "experiment": "text_groundingdino_sam",
                    "text_prompt": text_prompt,
                    "status": "no_text_box_found",
                    "groundingdino_score": None,
                    "text_box_iou_with_gt_box": 0.0,
                    "sam_score": None,
                    "mask_inside_gt_box": None,
                    "gt_box_covered_by_mask": None,
                    "pred_box_iou_with_gt_box": None,
                    "mask_area": None,
                })
                continue

            text_box_iou = box_iou(text_box, gt_box)

            # Step 2: Use GroundingDINO box as prompt for SAM segmentations
            mask_np, best_idx, sam_score = run_sam_with_box(
                image, text_box, sam_model, sam_processor, device
            )
            metrics = evaluate_mask_against_gt_box(mask_np, gt_box)

            rows.append({
                "image": image_name,
                "label": label,
                "experiment": "text_groundingdino_sam",
                "text_prompt": text_prompt,
                "status": "ok",
                "groundingdino_score": gd_score,
                "groundingdino_label": gd_label,
                "text_box_iou_with_gt_box": text_box_iou,
                "sam_score": sam_score,
                "best_mask_index": best_idx,
                "mask_inside_gt_box": metrics["mask_inside_gt_box"],
                "gt_box_covered_by_mask": metrics["gt_box_covered_by_mask"],
                "pred_box_iou_with_gt_box": metrics["pred_box_iou_with_gt_box"],
                "mask_area": metrics["mask_area"],
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
                "text_x1": text_box[0],
                "text_y1": text_box[1],
                "text_x2": text_box[2],
                "text_y2": text_box[3],
            })

            # Save visual overlay results up to max allowed examples
            if saved_count < args.save_examples:
                base = os.path.splitext(image_name)[0]
                save_text_overlay(
                    image=image,
                    gt_box=gt_box,
                    text_box=text_box,
                    mask_np=mask_np,
                    title=f"Text Prompt: {text_prompt}",
                    out_path=os.path.join(example_dir, f"{base}_text.png")
                )
                saved_count += 1

        except Exception as e:
            rows.append({
                "image": image_name,
                "label": label,
                "experiment": "text_groundingdino_sam",
                "status": "error",
                "error": str(e)
            })

    # Save detailed evaluation results to CSV
    result_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "text_prompt_results.csv")
    result_df.to_csv(out_csv, index=False)

    print("\nFinished.")
    print("CSV saved to:", out_csv)
    print("Examples saved in:", example_dir)

    print("\nStatus counts:")
    print(result_df["status"].value_counts())

    # Compute and save mean summary statistics for successful predictions
    ok_df = result_df[result_df["status"] == "ok"].copy()
    if len(ok_df) > 0:
        metrics = [
            "groundingdino_score",
            "text_box_iou_with_gt_box",
            "sam_score",
            "mask_inside_gt_box",
            "gt_box_covered_by_mask",
            "pred_box_iou_with_gt_box",
            "mask_area"
        ]

        summary = ok_df[metrics].mean()
        print("\nMean metrics for successful text detections:")
        print(summary)

        summary.to_csv(os.path.join(args.output_dir, "text_prompt_summary.csv"))


if __name__ == "__main__":
    main()