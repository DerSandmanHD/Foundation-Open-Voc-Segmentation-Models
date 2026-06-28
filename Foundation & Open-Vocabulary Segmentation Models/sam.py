import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.patches import Rectangle
from transformers import SamModel, SamProcessor


# ============================================================
# 0. Einstellungen
# ============================================================

image_name = "00013118_008.png"

# Arzt-Bounding-Box aus CSV
x = 225.084746
y = 547.019217
w = 86.779661
h = 79.186441

x_min = x
y_min = y
x_max = x + w
y_max = y + h
gt_box = [x_min, y_min, x_max, y_max]

# Für Experiment 4
# Falls die Diagnose aus eurer CSV anders ist, hier anpassen:
TEXT_PROMPT = "atelectasis."

# Für Experiment 3: automatische Suche über Punktgitter
# 5 = schneller, 8 oder 10 = gründlicher, aber langsamer
AUTO_GRID_SIZE = 5

OUTPUT_DIR = "sam_experimente_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nutze Gerät: {device}")


# ============================================================
# 1. Bild finden und laden
# ============================================================

image_path = None
print(f"Suche nach {image_name} ...")

for root, dirs, files in os.walk("."):
    if image_name in files:
        image_path = os.path.join(root, image_name)
        break

if image_path is None:
    raise FileNotFoundError(
        f"Konnte {image_name} nicht finden. Prüfe, ob die NIH-Bildordner im aktuellen Verzeichnis liegen."
    )

print(f"Bild gefunden: {image_path}")
image = Image.open(image_path).convert("RGB")
img_w, img_h = image.size
print("Bildgröße:", image.size)


# ============================================================
# 2. SAM laden
# ============================================================

print("Lade SAM ...")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model.eval()
print("SAM bereit.")


# ============================================================
# 3. Hilfsfunktionen: Metriken und Visualisierung
# ============================================================

def box_iou(box_a, box_b):
    """
    IoU zwischen zwei Bounding Boxes im Format [x1, y1, x2, y2].
    """
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
    if union == 0:
        return 0.0

    return inter_area / union


def mask_to_bbox(mask_bool):
    """
    Erzeugt aus einer Binärmaske eine Bounding Box.
    """
    ys, xs = np.where(mask_bool)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return [xs.min(), ys.min(), xs.max(), ys.max()]


def evaluate_mask_against_gt_box(mask_np, gt_box):
    """
    Achtung:
    Das ist keine echte medizinische Pixel-Dice-Auswertung,
    weil NIH hier meist nur Boxen und keine echten Pixelmasken liefert.

    Wir messen:
    - Wie viel der SAM-Maske liegt innerhalb der Arztbox?
    - Wie viel der Arztbox wird von der SAM-Maske abgedeckt?
    - Wie stark überlappt die aus der Maske berechnete Box mit der Arztbox?
    """
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
    bbox_iou = box_iou(pred_box, gt_box) if pred_box is not None else 0.0

    return {
        "mask_area": int(mask_area),
        "mask_inside_gt_box": float(mask_inside_gt),
        "gt_box_covered_by_mask": float(gt_covered_by_mask),
        "pred_box": pred_box,
        "pred_box_iou_with_gt_box": float(bbox_iou),
    }


def get_best_sam_mask(outputs, inputs):
    """
    SAM gibt meist 3 Masken zurück.
    Wir wählen die mit dem höchsten SAM-internen IoU-Score.
    """
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    # Für ein Bild und einen Prompt:
    # masks[0] hat typischerweise Shape: [1, 3, H, W]
    scores = outputs.iou_scores.detach().cpu()

    best_idx = scores[0, 0].argmax().item()
    best_score = scores[0, 0, best_idx].item()

    mask_np = masks[0][0][best_idx].numpy()

    return mask_np, best_idx, best_score


def plot_experiment(result, filename):
    """
    Einzelbild pro Experiment speichern.
    """
    mask_np = result.get("mask")
    title = result.get("title", "Experiment")
    prompt_box = result.get("prompt_box")
    prompt_point = result.get("prompt_point")
    text_box = result.get("text_box")
    error = result.get("error")

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)

    # Ground-Truth-/Arztbox immer gestrichelt anzeigen
    rect_gt = Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label="Arztbox / Ground Truth"
    )
    ax.add_patch(rect_gt)

    # Prompt-Box, falls vorhanden
    if prompt_box is not None:
        rect_prompt = Rectangle(
            (prompt_box[0], prompt_box[1]),
            prompt_box[2] - prompt_box[0],
            prompt_box[3] - prompt_box[1],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="-",
            label="Prompt-Box"
        )
        ax.add_patch(rect_prompt)

    # Text-Box aus GroundingDINO, falls vorhanden
    if text_box is not None:
        rect_text = Rectangle(
            (text_box[0], text_box[1]),
            text_box[2] - text_box[0],
            text_box[3] - text_box[1],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="-",
            label="Text-Modell-Box"
        )
        ax.add_patch(rect_text)

    # Prompt-Punkt, falls vorhanden
    if prompt_point is not None:
        ax.scatter([prompt_point[0]], [prompt_point[1]], s=80, marker="x", c="yellow", label="Point-Prompt")

    # SAM-Maske
    if mask_np is not None:
        mask_show = np.where(mask_np > 0.5, mask_np, np.nan)
        ax.imshow(mask_show, cmap="cool", alpha=0.5)

    if error is not None:
        ax.text(
            20, 40,
            error,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.7)
        )

    ax.legend(loc="lower right")

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

    print(f"Gespeichert: {out_path}")


def print_result_metrics(result):
    print("\n" + "=" * 70)
    print(result["title"])
    print("=" * 70)

    if result.get("error"):
        print("Fehler / Ergebnis:", result["error"])
        return

    print("SAM best mask index:", result.get("best_mask_index"))
    print("SAM interner IoU-Score:", result.get("sam_score"))

    metrics = result.get("metrics")
    if metrics:
        print("Maskenfläche:", metrics["mask_area"])
        print("Anteil der Maske innerhalb Arztbox:", round(metrics["mask_inside_gt_box"], 4))
        print("Anteil der Arztbox durch Maske abgedeckt:", round(metrics["gt_box_covered_by_mask"], 4))
        print("Box-IoU zwischen Masken-Box und Arztbox:", round(metrics["pred_box_iou_with_gt_box"], 4))
        print("Aus Maske berechnete Box:", metrics["pred_box"])


# ============================================================
# 4. Experiment 1: SAM mit Bounding Box
# ============================================================

def run_sam_with_box(box, title):
    input_box = [[box]]  # Format: [Bild][Box][x1,y1,x2,y2]

    inputs = sam_processor(
        image,
        input_boxes=input_box,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask_np, best_idx, sam_score = get_best_sam_mask(outputs, inputs)
    metrics = evaluate_mask_against_gt_box(mask_np, gt_box)

    return {
        "title": title,
        "mask": mask_np,
        "prompt_box": box,
        "prompt_point": None,
        "best_mask_index": best_idx,
        "sam_score": sam_score,
        "metrics": metrics,
    }


exp1 = run_sam_with_box(
    gt_box,
    "Experiment 1: SAM mit Arzt-Bounding-Box"
)

print_result_metrics(exp1)
plot_experiment(exp1, "exp1_box_prompt.png")


# ============================================================
# 5. Experiment 2: SAM mit Point-Prompt
# ============================================================

def run_sam_with_point(point, title):
    px, py = point

    input_points = [[[px, py]]]  # Format: [Bild][Punkt][x,y]

    inputs = sam_processor(
        image,
        input_points=input_points,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask_np, best_idx, sam_score = get_best_sam_mask(outputs, inputs)
    metrics = evaluate_mask_against_gt_box(mask_np, gt_box)

    return {
        "title": title,
        "mask": mask_np,
        "prompt_box": None,
        "prompt_point": point,
        "best_mask_index": best_idx,
        "sam_score": sam_score,
        "metrics": metrics,
    }


point_x = x_min + w / 2
point_y = y_min + h / 2
center_point = [point_x, point_y]

exp2 = run_sam_with_point(
    center_point,
    "Experiment 2: SAM mit Mittelpunkt der Arztbox"
)

print_result_metrics(exp2)
plot_experiment(exp2, "exp2_point_prompt.png")


# ============================================================
# 6. Experiment 3: SAM ohne Arztbox
#    Einfache automatische Suche über Punktgitter
# ============================================================

def run_auto_grid_search():
    """
    Hier bekommt SAM NICHT die Arztbox.
    Wir geben viele gleichmäßig verteilte Punkte über das Bild.

    Wichtig:
    Die Arztbox wird nur nachträglich zur Bewertung benutzt.
    Für die Visualisierung wählen wir den Kandidaten, der am besten mit der Arztbox überlappt.
    Das ist eine Analysefrage:
    'War unter den automatisch erzeugten Kandidaten überhaupt etwas Passendes dabei?'
    """
    print("\nStarte Experiment 3: automatische Punktgitter-Suche ...")

    xs = np.linspace(0.1 * img_w, 0.9 * img_w, AUTO_GRID_SIZE)
    ys = np.linspace(0.1 * img_h, 0.9 * img_h, AUTO_GRID_SIZE)

    candidates = []

    for gy in ys:
        for gx in xs:
            candidate = run_sam_with_point(
                [float(gx), float(gy)],
                "Auto-Kandidat"
            )

            candidate["grid_point"] = [float(gx), float(gy)]
            candidates.append(candidate)

    # Kandidat mit höchster Übereinstimmung zur Arztbox.
    # Achtung: Das ist keine Modellentscheidung, sondern nachträgliche Analyse.
    best_candidate = max(
        candidates,
        key=lambda r: r["metrics"]["pred_box_iou_with_gt_box"]
    )

    best_candidate["title"] = (
        "Experiment 3: SAM ohne Arztbox\n"
        "Bester Auto-Kandidat nachträglich gegen Arztbox ausgewählt"
    )
    best_candidate["prompt_point"] = best_candidate["grid_point"]

    return best_candidate, candidates


exp3, auto_candidates = run_auto_grid_search()

print_result_metrics(exp3)
plot_experiment(exp3, "exp3_auto_grid_best_candidate.png")


# ============================================================
# 7. Experiment 4: Text/Open-Vocabulary mit GroundingDINO + SAM
# ============================================================

def run_text_prompt_groundingdino_sam(text_prompt):
    """
    GroundingDINO versucht aus Text eine Box zu finden.
    Diese Box wird dann als Prompt für SAM benutzt.

    Erwartung bei medizinischen Röntgenbildern:
    Das kann gut fehlschlagen. Genau dieser Failure Mode ist interessant.
    """
    print("\nStarte Experiment 4: Text/Open-Vocabulary ...")
    print("Text-Prompt:", text_prompt)

    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        gd_model_id = "IDEA-Research/grounding-dino-tiny"

        print("Lade GroundingDINO ...")
        gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(device)
        gd_model.eval()

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

        target_sizes = torch.tensor([[img_h, img_w]], device=device)

        try:
            results = gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.20,
                text_threshold=0.20,
                target_sizes=[image.size[::-1]]
            )[0]
        except TypeError:
            results = gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.20,
                text_threshold=0.20,
                target_sizes=[image.size[::-1]]
            )[0]

        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["labels"]

        if len(boxes) == 0:
            return {
                "title": f"Experiment 4: Text-Prompt '{text_prompt}'",
                "mask": None,
                "text_box": None,
                "prompt_box": None,
                "prompt_point": None,
                "error": "Keine Text-Box gefunden.\nDas ist ein möglicher Open-Vocabulary-Failure.",
            }

        best_id = int(np.argmax(scores))
        text_box = boxes[best_id].tolist()
        text_score = float(scores[best_id])
        text_label = labels[best_id]

        print("Gefundene Text-Box:", text_box)
        print("GroundingDINO Score:", text_score)
        print("GroundingDINO Label:", text_label)

        # SAM segmentiert die von GroundingDINO gefundene Box
        sam_result = run_sam_with_box(
            text_box,
            f"Experiment 4: Text → GroundingDINO-Box → SAM\nPrompt: {text_prompt}"
        )

        sam_result["text_box"] = text_box
        sam_result["groundingdino_score"] = text_score
        sam_result["groundingdino_label"] = text_label

        return sam_result

    except Exception as e:
        return {
            "title": f"Experiment 4: Text-Prompt '{text_prompt}'",
            "mask": None,
            "text_box": None,
            "prompt_box": None,
            "prompt_point": None,
            "error": (
                "Experiment 4 konnte nicht ausgeführt werden.\n"
                "Mögliche Gründe: Transformers-Version zu alt,\n"
                "GroundingDINO nicht verfügbar oder Downloadproblem.\n\n"
                f"Fehler: {str(e)[:250]}"
            ),
        }


exp4 = run_text_prompt_groundingdino_sam(TEXT_PROMPT)

print_result_metrics(exp4)
plot_experiment(exp4, "exp4_text_groundingdino_sam.png")


# ============================================================
# 8. Gesamtübersicht aller Experimente
# ============================================================

def plot_overview(results, filename="overview_all_experiments.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))

    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(result["title"], fontsize=10)

        # Arztbox als Referenz
        rect_gt = Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--"
        )
        ax.add_patch(rect_gt)

        # Prompt Box
        if result.get("prompt_box") is not None:
            b = result["prompt_box"]
            rect_prompt = Rectangle(
                (b[0], b[1]),
                b[2] - b[0],
                b[3] - b[1],
                linewidth=2,
                edgecolor="yellow",
                facecolor="none"
            )
            ax.add_patch(rect_prompt)

        # Text Box
        if result.get("text_box") is not None:
            b = result["text_box"]
            rect_text = Rectangle(
                (b[0], b[1]),
                b[2] - b[0],
                b[3] - b[1],
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )
            ax.add_patch(rect_text)

        # Prompt Point
        if result.get("prompt_point") is not None:
            p = result["prompt_point"]
            ax.scatter([p[0]], [p[1]], s=80, marker="x", c="yellow")

        # Maske
        if result.get("mask") is not None:
            mask_show = np.where(result["mask"] > 0.5, result["mask"], np.nan)
            ax.imshow(mask_show, cmap="cool", alpha=0.5)

        # Fehlertext
        if result.get("error") is not None:
            ax.text(
                20, 40,
                result["error"],
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.75)
            )

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

    print(f"\nGesamtübersicht gespeichert: {out_path}")


plot_overview([exp1, exp2, exp3, exp4])


# ============================================================
# 9. Kurze Zusammenfassung als Tabelle
# ============================================================

summary_rows = []

for name, result in [
    ("Box-Prompt", exp1),
    ("Point-Prompt", exp2),
    ("Auto-SAM ohne Arztbox", exp3),
    ("Text/Open-Vocabulary", exp4),
]:
    if result.get("metrics") is not None:
        m = result["metrics"]
        summary_rows.append({
            "Experiment": name,
            "SAM Score": result.get("sam_score"),
            "Mask inside GT box": m["mask_inside_gt_box"],
            "GT box covered by mask": m["gt_box_covered_by_mask"],
            "Pred-box IoU with GT box": m["pred_box_iou_with_gt_box"],
            "Mask area": m["mask_area"],
        })
    else:
        summary_rows.append({
            "Experiment": name,
            "SAM Score": None,
            "Mask inside GT box": None,
            "GT box covered by mask": None,
            "Pred-box IoU with GT box": None,
            "Mask area": None,
        })

try:
    import pandas as pd

    df_summary = pd.DataFrame(summary_rows)
    print("\nZusammenfassung:")
    display(df_summary)

    csv_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"CSV gespeichert: {csv_path}")

except Exception:
    print("\nZusammenfassung:")
    for row in summary_rows:
        print(row)


print("\nFertig. Alle Bilder liegen im Ordner:", OUTPUT_DIR)