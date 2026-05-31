import os
import pandas as pd
import matplotlib.pyplot as plt

SAM_CSV = r"sam_batch_output\sam_batch_results.csv"
TEXT_CSV = r"text_prompt_output_50\text_prompt_results.csv"

OUT_DIR = "combined_plots"
os.makedirs(OUT_DIR, exist_ok=True)

sam_df = pd.read_csv(SAM_CSV)
text_df = pd.read_csv(TEXT_CSV)

sam_df = sam_df[sam_df["status"] == "ok"].copy()
text_df = text_df[text_df["status"] == "ok"].copy()

common_cols = [
    "image",
    "label",
    "experiment",
    "status",
    "sam_score",
    "mask_inside_gt_box",
    "gt_box_covered_by_mask",
    "pred_box_iou_with_gt_box",
    "mask_area"
]

# Concatenate successful results from both experimental setups
combined = pd.concat(
    [
        sam_df[common_cols],
        text_df[common_cols]
    ],
    ignore_index=True
)

combined.to_csv(os.path.join(OUT_DIR, "combined_results.csv"), index=False)

metrics = [
    "sam_score",
    "mask_inside_gt_box",
    "gt_box_covered_by_mask",
    "pred_box_iou_with_gt_box",
    "mask_area"
]

summary_mean = combined.groupby("experiment")[metrics].mean()
summary_median = combined.groupby("experiment")[metrics].median()

summary_mean.to_csv(os.path.join(OUT_DIR, "combined_summary_mean.csv"))
summary_median.to_csv(os.path.join(OUT_DIR, "combined_summary_median.csv"))

print("\nMean summary:")
print(summary_mean)

print("\nMedian summary:")
print(summary_median)

# Barplots: Comparative mean values
for metric in metrics:
    plt.figure(figsize=(9, 5))
    ax = summary_mean[metric].plot(kind="bar")

    plt.ylabel(f"Mean {metric}")
    plt.title(f"Mean {metric} by prompt type")
    plt.xticks(rotation=20)
    plt.tight_layout()

    # Annotate values above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)

    plt.savefig(os.path.join(OUT_DIR, f"bar_mean_{metric}.png"), dpi=300)
    plt.close()

# Boxplots: Performance distribution across experiments
for metric in metrics:
    plt.figure(figsize=(9, 5))

    experiments = list(combined["experiment"].unique())
    data = [
        combined[combined["experiment"] == exp][metric].dropna()
        for exp in experiments
    ]

    plt.boxplot(data, labels=experiments)
    plt.ylabel(metric)
    plt.title(f"Distribution of {metric}")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, f"boxplot_{metric}.png"), dpi=300)
    plt.close()

# Additional: Plot GroundingDINO bounding box IoU standalone if available
if "text_box_iou_with_gt_box" in text_df.columns:
    plt.figure(figsize=(6, 5))
    text_df["text_box_iou_with_gt_box"].plot(kind="box")
    plt.ylabel("Text Box IoU with GT Box")
    plt.title("GroundingDINO Text Box IoU with GT Box")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "boxplot_groundingdino_text_box_iou.png"), dpi=300)
    plt.close()

print("\nPlots saved in:", OUT_DIR)