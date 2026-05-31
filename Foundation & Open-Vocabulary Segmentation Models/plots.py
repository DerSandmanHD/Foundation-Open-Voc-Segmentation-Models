import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "sam_batch_output"
CSV_PATH = os.path.join(RESULT_DIR, "sam_batch_results.csv")
OUT_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

print("Reading:", CSV_PATH)

df = pd.read_csv(CSV_PATH)
df = df[df["status"] == "ok"].copy()

metrics = [
    "sam_score",
    "mask_inside_gt_box",
    "gt_box_covered_by_mask",
    "pred_box_iou_with_gt_box",
    "mask_area"
]

summary = df.groupby("experiment")[metrics].mean()
print("\nSummary:")
print(summary)

summary.to_csv(os.path.join(OUT_DIR, "summary_for_plots.csv"))

# Barplots: Mean values
for metric in metrics:
    plt.figure(figsize=(7, 5))
    ax = summary[metric].plot(kind="bar")

    plt.ylabel(f"Mean {metric}")
    plt.title(f"Mean {metric} by prompt type")
    plt.xticks(rotation=20)
    plt.tight_layout()

    # Annotate values above the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)

    out_path = os.path.join(OUT_DIR, f"bar_{metric}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

# Boxplots: Distribution variations
for metric in metrics:
    plt.figure(figsize=(7, 5))

    experiments = list(df["experiment"].unique())
    data = [
        df[df["experiment"] == exp][metric].dropna()
        for exp in experiments
    ]

    plt.boxplot(data, labels=experiments)
    plt.ylabel(metric)
    plt.title(f"Distribution of {metric}")
    plt.xticks(rotation=20)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"boxplot_{metric}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

print("\nPlots saved in:", OUT_DIR)