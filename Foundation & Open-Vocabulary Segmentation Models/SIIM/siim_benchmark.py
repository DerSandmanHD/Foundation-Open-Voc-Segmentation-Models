import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
from scipy.ndimage import label

# ============================================================
# 1. Path Configuration
# ============================================================
IMG_DIR = r"archive\input\input\train\images\1024\dicom"
MASK_DIR = r"archive\input\input\train\images\1024\mask"
OUT_DIR = "siim_benchmark_final_output"
MAX_IMAGES = 20  # For testing purposes

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. Helper Functions (Metrics & Bounding Boxes)
# ============================================================
def mask_to_multiple_bboxes(mask_np):
    """Separates the mask into individual instances and calculates bounding boxes."""
    labeled_mask, num_features = label(mask_np > 0)
    bboxes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled_mask == i)
        if len(xs) > 0 and len(ys) > 0:
            bboxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return bboxes

def get_bbox_from_mask(mask_np):
    """Extracts a single bounding box from a predicted mask for IoU comparison."""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def dice_score(pred, gt):
    """Calculates the pixel-level Dice score."""
    pred, gt = pred > 0, gt > 0
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    return float(2.0 * intersection / denominator) if denominator > 0 else 1.0

def calculate_iou(box1, box2):
    """Calculates the Box-IoU for comparability with the NIH dataset."""
    if box1 is None or box2 is None: return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return float(inter_area) / float(box1_area + box2_area - inter_area)

def eval_metrics(pred_box, gt_box, iou_thresh=0.5):
    """Checks if the IoU is large enough for a True Positive (Recall)."""
    iou = calculate_iou(pred_box, gt_box)
    tp = 1 if iou >= iou_thresh else 0
    return iou, tp

# ============================================================
# 3. Main Program
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.png')]
    valid_pairs = []
    for f in all_images:
        mask_path = os.path.join(MASK_DIR, f)
        if os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path).convert("L"))
            if mask_img.sum() > 0:
                valid_pairs.append((os.path.join(IMG_DIR, f), mask_path, f))
        if len(valid_pairs) >= MAX_IMAGES:
            break

    print(f"\nLoading {len(valid_pairs)} images for the final benchmark...")

    print("Loading SAM 1 (Facebook)...")
    processor_sam1 = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model_sam1 = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    model_sam1.eval()

    print("Loading MedSAM (WangLab)...")
    processor_medsam = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
    model_medsam = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
    model_medsam.eval()

    # Tracking lists for results
    dice_sam1_list, dice_medsam_list = [], []
    iou_sam1_list, iou_medsam_list = [], []
    tp_sam1, tp_medsam = 0, 0
    total_boxes = 0

    for img_path, mask_path, filename in tqdm(valid_pairs, desc="Running Benchmark"):
        image = Image.open(img_path).convert("RGB")
        gt_mask = np.array(Image.open(mask_path).convert("L")) > 0
        
        # Calculate Ground Truth Boxes
        gt_boxes = mask_to_multiple_bboxes(gt_mask)
        if not gt_boxes: continue

        total_boxes += len(gt_boxes)

        # If testing with DINO, call the DINO function here:
        # prompt_boxes = get_dino_boxes(image, "pneumothorax") 
        # And use 'prompt_boxes' instead of 'gt_boxes' below.
        prompt_boxes = gt_boxes 

        pred_mask_sam1 = np.zeros(gt_mask.shape, dtype=bool)
        pred_mask_medsam = np.zeros(gt_mask.shape, dtype=bool)

        for box in prompt_boxes:
            # --- SAM 1 Prediction ---
            inputs_s1 = processor_sam1(image, input_boxes=[[box]], return_tensors="pt").to(device)
            with torch.no_grad():
                out_s1 = model_sam1(**inputs_s1)
            masks_s1 = processor_sam1.image_processor.post_process_masks(
                out_s1.pred_masks.cpu(), inputs_s1["original_sizes"].cpu(), inputs_s1["reshaped_input_sizes"].cpu()
            )
            best_idx_s1 = out_s1.iou_scores.detach().cpu()[0, 0].argmax().item()
            single_pred_s1 = masks_s1[0][0][best_idx_s1].numpy() > 0.5
            pred_mask_sam1 = np.logical_or(pred_mask_sam1, single_pred_s1)
            
            # Box-IoU Evaluation for SAM 1
            pred_box_s1 = get_bbox_from_mask(single_pred_s1)
            if pred_box_s1:
                iou, tp = eval_metrics(pred_box_s1, box)
                iou_sam1_list.append(iou)
                tp_sam1 += tp

            # --- MedSAM Prediction ---
            inputs_med = processor_medsam(image, input_boxes=[[box]], return_tensors="pt").to(device)
            with torch.no_grad():
                out_med = model_medsam(**inputs_med)
            masks_med = processor_medsam.image_processor.post_process_masks(
                out_med.pred_masks.cpu(), inputs_med["original_sizes"].cpu(), inputs_med["reshaped_input_sizes"].cpu()
            )
            best_idx_med = out_med.iou_scores.detach().cpu()[0, 0].argmax().item()
            single_pred_med = masks_med[0][0][best_idx_med].numpy() > 0.5
            pred_mask_medsam = np.logical_or(pred_mask_medsam, single_pred_med)

            # Box-IoU Evaluation for MedSAM
            pred_box_med = get_bbox_from_mask(single_pred_med)
            if pred_box_med:
                iou, tp = eval_metrics(pred_box_med, box)
                iou_medsam_list.append(iou)
                tp_medsam += tp

        # Calculate Pixel-Dice Metrics (Image-Level)
        dice_s1 = dice_score(pred_mask_sam1, gt_mask)
        dice_med = dice_score(pred_mask_medsam, gt_mask)
        dice_sam1_list.append(dice_s1)
        dice_medsam_list.append(dice_med)

        # ============================================================
        # Benchmark Visualization (4 Columns)
        # ============================================================
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image)
        for box in prompt_boxes:
            x1, y1, x2, y2 = box
            axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2))
        axes[0].set_title(f"Input ({len(prompt_boxes)} Prompts)")
        axes[0].axis("off")
        
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth (Experts)")
        axes[1].axis("off")
        
        axes[2].imshow(pred_mask_sam1, cmap="Reds", alpha=0.5)
        axes[2].imshow(image, cmap="gray", alpha=0.5)
        axes[2].set_title(f"SAM 1 (Dice: {dice_s1:.4f})")
        axes[2].axis("off")

        axes[3].imshow(pred_mask_medsam, cmap="Greens", alpha=0.5)
        axes[3].imshow(image, cmap="gray", alpha=0.5)
        axes[3].set_title(f"MedSAM (Dice: {dice_med:.4f})")
        axes[3].axis("off")
        
        plt.tight_layout()
        save_name = os.path.join(OUT_DIR, f"compare_{os.path.splitext(filename)[0]}.png")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()

    # ============================================================
    # Console Output of Final Metrics
    # ============================================================
    if total_boxes > 0:
        recall_s1 = tp_sam1 / total_boxes
        recall_med = tp_medsam / total_boxes
        
        print("\n" + "="*60)
        print("📊 FINAL BENCHMARK RESULTS (SIIM Pneumothorax)")
        print("="*60)
        print("🔹 MODEL: SAM 1 (Base)")
        print(f"   Pixel-Level: Mean Dice-Score = {np.mean(dice_sam1_list):.4f}")
        print(f"   Box-Level:   Mean Box-IoU    = {np.mean(iou_sam1_list):.4f}" if iou_sam1_list else "   Box-Level:   Mean Box-IoU    = N/A")
        print(f"   Box-Level:   Recall@50       = {recall_s1:.4f}")
        print("-" * 60)
        print("🔹 MODEL: MedSAM (Medical)")
        print(f"   Pixel-Level: Mean Dice-Score = {np.mean(dice_medsam_list):.4f}")
        print(f"   Box-Level:   Mean Box-IoU    = {np.mean(iou_medsam_list):.4f}" if iou_medsam_list else "   Box-Level:   Mean Box-IoU    = N/A")
        print(f"   Box-Level:   Recall@50       = {recall_med:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()