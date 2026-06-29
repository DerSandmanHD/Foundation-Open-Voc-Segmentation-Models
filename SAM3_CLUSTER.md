# SAM3 in der bestehenden TCML-Singularity-Struktur

Die SAM3-Dateien sind nach demselben Muster aufgebaut wie das vorhandene
Project-Aria-Setup:

```text
~/singularity/
├── sam3.recipe
├── sam3_master.simg                 # entsteht beim Build
├── sam3_download_checkpoint.sh
├── sam3_smoke.sbatch
├── sam3_nih.sbatch
├── models/
│   ├── sam3.1_multiplex.pt          # separater, zugriffsgeschützter Checkpoint
│   └── config.json
├── sam3_smoke.JOBID.out
├── sam3_smoke.JOBID.err
├── sam3_nih.JOBID.out
└── sam3_nih.JOBID.err
```

## Was bereits im Projekt liegt

Im hochgeladenen Projekt liegt SAM3-Quellcode, aber nicht das vollständige
Modell:

1. `sam3.1_multiplex.pt` beziehungsweise andere Modellgewichte sind nicht vorhanden.
2. Der Quellcode ist unvollständig. Insbesondere fehlt
   `sam3/sam3/train/data/collator.py`, obwohl die Inferenzmodule diese Datei
   importieren. Ursache war der globale `.gitignore`-Eintrag `data/`.

Das Rezept installiert deshalb einen vollständigen, gepinnten Checkout des
offiziellen SAM3-Repositories in das Image. Die zugriffsgeschützten Gewichte
werden nach dem Build separat über dein Hugging-Face-Konto heruntergeladen.

## 1. Dateien im Projekt verwenden

Auf login3 kann direkt der bereits vorhandene Projektordner verwendet werden:

```bash
cd ~/projekte/Foundation-Open-Voc-Segmentation-Models/singularity
```

Die Batchdateien leiten die Projektwurzel automatisch vom Verzeichnis ab, in
dem `sbatch` aufgerufen wird. Falls die Singularity-Dateien separat nach
`~/singularity` kopiert werden, muss `SAM3_PROJECT_ROOT` explizit gesetzt werden.

## 2. Image bauen

```bash
cd ~/projekte/Foundation-Open-Voc-Segmentation-Models/singularity
singularity build --fakeroot sam3_master.simg sam3.recipe
```

Das Rezept enthält:

- Ubuntu 24.04 und CUDA 12.8.1
- Python 3.12 in `/opt/sam3_env`
- PyTorch 2.10.0 und Torchvision 0.25.0 mit CUDA 12.8
- Setuptools 80.9.0, da SAM3 noch `pkg_resources` verwendet
- explizit ergänzte SAM3-Laufzeitimporte `einops`, `psutil` und `pycocotools`,
  die in den offiziellen Basis-Abhängigkeiten nicht vollständig deklariert sind
- vollständigen SAM3-Commit `8e451d5`
- NumPy, Pandas, Matplotlib, Pillow und tqdm für den Benchmark

Das Image kann anschließend geprüft werden:

```bash
singularity inspect sam3_master.simg
singularity test sam3_master.simg
```

Die CUDA-Verfügbarkeit ist während des Builds normalerweise `False`, weil der
Build auf dem Login-Knoten ohne GPU läuft. Entscheidend ist später
`singularity exec --nv` innerhalb eines Slurm-GPU-Jobs.

## 3. Hugging-Face-Zugriff und Checkpoint

Sobald dein Antrag für `facebook/sam3.1` genehmigt wurde:

```bash
cd ~/projekte/Foundation-Open-Voc-Segmentation-Models/singularity
singularity exec sam3_master.simg hf auth login
bash sam3_download_checkpoint.sh
```

Danach müssen diese Dateien existieren:

```bash
ls -lh models/sam3.1_multiplex.pt
ls -lh models/config.json
```

Der Checkpoint wird nicht in das Image eingebaut. Dadurch bleibt dein
Hugging-Face-Token außerhalb des Images und der Checkpoint kann unabhängig vom
Container ausgetauscht werden.

## 4. Smoke-Test

```bash
cd ~/projekte/Foundation-Open-Voc-Segmentation-Models/singularity
sbatch sam3_smoke.sbatch \
  --prompt "truck"
```

Ohne `--image` verwenden die Smoke-Jobs automatisch das eingecheckte
`truck.jpg`. Der Pfad wird aus `SLURM_SUBMIT_DIR` abgeleitet und funktioniert
damit auch dann, wenn TCML das Home-Verzeichnis unter `/mnt/beegfs/home/...`
statt `/home/...` einbindet. Ein eigenes Bild kann weiterhin mit `--image`
angegeben werden; da zusätzliche Argumente zuletzt stehen, überschreibt es das
Standardbild.

Beide Smoke-Jobs verwenden die `day`-Partition und eine A4000. Die im
SAM3-Container installierten PyTorch-2.10-/CUDA-12.8-Wheels unterstützen erst
CUDA Compute Capability 7.0; die GTX 1080 Ti besitzt nur 6.1 und kann diese
Wheels nicht ausführen. Der SAM3.1-Multiplex-Predictor behandelt das einzelne
Bild intern als Ein-Frame-Sequenz. Logs werden als `sam3_smoke.JOBID.out` und
`sam3_smoke.JOBID.err` gespeichert.

Wenn das Projekt auf dem Cluster anders heißt:

```bash
export SAM3_PROJECT_ROOT=/home/eker/PFAD/ZUM/PROJEKT
sbatch sam3_smoke.sbatch --image /home/eker/pfad/testbild.png --prompt "lungs"
```

## 5. NIH-Textbenchmark

Der Job verwendet standardmäßig den normalen SAM3-Checkpoint `sam3.pt`, also
dasselbe Modell und dieselbe FP16-Inferenz wie der erfolgreiche
`sam3_smoke.sbatch`-Lauf. Die `BBox_List_2017.csv` enthält 984 räumliche
Annotationen für 880 eindeutige Bilder. Ohne `--label` und ohne
`--max-annotations` werden alle 984 Annotationen über alle acht Pathologien
ausgewertet.

Mit der Verzeichnisstruktur `~/projekte/data/NIH_Dataset` genügt:

```bash
cd ~/projekte/Foundation-Open-Voc-Segmentation-Models/singularity
sbatch sam3_nih.sbatch
```

Der Benchmark fordert eine A4000 in der `day`-Partition an und verwendet FP16.
Er speichert keine Visualisierungen, sondern die Einzelresultate, eine
Gesamtzusammenfassung und eine Zusammenfassung je Pathologie unter
`$SAM3_PROJECT_ROOT/sam3_outputs/nih_all_bbox_annotations`.

Für einen kurzen Testlauf mit zehn gespeicherten Beispielen können Argumente
überschrieben werden:

```bash
sbatch sam3_nih.sbatch --max-annotations 50 --save-examples 10
```

Ein Filter wie `--label Atelectasis` wertet nur diese Pathologie aus. Ein
festes `--text-prompt` sollte beim Lauf über alle Pathologien nicht gesetzt
werden; standardmäßig wird pro Annotation deren Pathologiebezeichnung als
Prompt verwendet.

Abweichende Pfade können über Umgebungsvariablen gesetzt werden:

```bash
export SAM3_NIH_ROOT=/anderer/pfad/NIH_Dataset
export SAM3_NIH_BBOX_CSV=/anderer/pfad/BBox_List_2017.csv
export SAM3_NIH_OUTPUT_DIR=/anderer/pfad/ergebnisse
sbatch sam3_nih.sbatch
```

Optional kann `SAM3_STAGE_DATASET` gesetzt werden, um die Bilder vor der
Inferenz nach `/scratch/$SLURM_JOB_ID` zu kopieren. Dafür muss ausreichend
lokaler Scratch-Speicher vorhanden sein.

## 6. SIIM-Pneumothorax-Textbenchmark

Der SIIM-Job sucht unter `~/projekte/data/SIIM_Dataset` automatisch nach dem
vorverarbeiteten Ordnerpaar `dicom`/`mask`. Er wertet alle Fälle mit nicht
leerer Pneumothorax-Maske aus und vereinigt alle SAM3-Textdetektionen eines
Bildes zu einer Vorhersagemaske.

```bash
cd ~/projekte/Foundation-Open-Voc-Segmentation-Models/singularity
export SAM3_SIIM_ROOT=/home/eker/projekte/data/SIIM_Dataset
sbatch sam3_siim.sbatch
```

Der Job speichert standardmäßig zehn mit festem Seed ausgewählte
Visualisierungen sowie Dice, Mask-IoU, Pixel-Precision und Pixel-Recall:

```text
sam3_outputs/siim_pneumothorax/
├── examples/
├── sam3_siim_results.csv
└── sam3_siim_summary.csv
```

Die Zahl der Präsentationsbilder oder ein kurzer Testlauf können über
Argumente gesteuert werden:

```bash
sbatch sam3_siim.sbatch --max-images 50 --save-examples 6
```

Falls die automatische Erkennung nicht zur lokalen Struktur passt, können die
beiden Ordner explizit übergeben werden:

```bash
sbatch sam3_siim.sbatch \
  --image-dir /pfad/zu/dicom \
  --mask-dir /pfad/zu/mask
```
