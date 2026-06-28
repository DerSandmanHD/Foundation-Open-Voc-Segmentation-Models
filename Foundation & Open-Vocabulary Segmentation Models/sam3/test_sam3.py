import os
import matplotlib.pyplot as plt
from PIL import Image

# SAM 3 Imports (aus deinem Snippet)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============================================================
# Pfad zu einem deiner SIIM Lungen-Bilder
# ============================================================
# ============================================================
# Pfad zu einem deiner SIIM Lungen-Bilder (ABSOLUTER PFAD)
# ============================================================
IMG_DIR = r"D:\Foundation-Open-Voc-Segmentation-Models\Foundation & Open-Vocabulary Segmentation Models\SIIM\archive\input\input\train\images\1024\dicom"
def main():
    # Erstes Bild aus dem Ordner schnappen
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")]
    if not images:
        print("Keine Bilder im Ordner gefunden!")
        return
    
    img_path = os.path.join(IMG_DIR, images[0])
    image = Image.open(img_path).convert("RGB")
    print(f"Teste SAM 3 mit Bild: {images[0]}")

    # 1. Modell laden
    print("Lade SAM 3 Modell...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # 2. Bild an das Modell übergeben
    print("Übergebe Bild an Processor...")
    inference_state = processor.set_image(image)

    # 3. TEXT-PROMPT TESTEN!
    # Wir probieren erst etwas Leichtes (Lunge) 
    prompt_text = "lungs"
    print(f"Suche nach: '{prompt_text}'...")
    output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

    masks = output["masks"]
    scores = output["scores"]

    # 4. Ergebnis visualisieren und speichern
    plt.figure(figsize=(10, 5))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Röntgenbild")
    plt.axis("off")
    
    # SAM 3 Maske
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    if len(masks) > 0:
        # Wir nehmen die Maske mit der höchsten Confidence
        best_mask = masks[0]
        
        # Falls es ein PyTorch Tensor ist, ins Numpy-Format wandeln
        if hasattr(best_mask, 'cpu'):
            best_mask = best_mask.squeeze().cpu().numpy()
            
        plt.imshow(best_mask, cmap="jet", alpha=0.5)
        
        # Score kann auch ein Tensor sein
        score_val = scores[0].item() if hasattr(scores[0], 'item') else scores[0]
        plt.title(f"SAM 3 Prompt: '{prompt_text}'\nScore: {score_val:.2f}")
    else:
        plt.title("SAM 3 hat nichts gefunden!")
    plt.axis("off")
    
    save_path = "sam3_test_result.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Fertig! Ergebnisbild gespeichert unter: {save_path}")

if __name__ == "__main__":
    main()