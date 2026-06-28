print("Initialisiere Kaggle API...")

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
except Exception as e:
    print(f"\n[FEHLER] Authentifizierung fehlgeschlagen: {e}")
    exit()

print("\nStarte Download von SIIM-ACR Pneumothorax Segmentation...")

try:
    api.competition_download_files(
        "siim-acr-pneumothorax-segmentation",
        path=".",
        force=False,
        quiet=False
    )
    print("\n[ERFOLG] Download abgeschlossen.")
    print("Jetzt ZIP-Datei entpacken.")
except Exception as e:
    print(f"\n[FEHLER] Download fehlgeschlagen: {e}")