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
│   ├── sam3.pt                      # separater, zugriffsgeschützter Checkpoint
│   └── config.json
├── sam3_smoke.JOBID.out
├── sam3_smoke.JOBID.err
├── sam3_nih.JOBID.out
└── sam3_nih.JOBID.err
```

## Was bereits im Projekt liegt

Im hochgeladenen Projekt liegt SAM3-Quellcode, aber nicht das vollständige
Modell:

1. `sam3.pt` beziehungsweise andere Modellgewichte sind nicht vorhanden.
2. Der Quellcode ist unvollständig. Insbesondere fehlt
   `sam3/sam3/train/data/collator.py`, obwohl die Inferenzmodule diese Datei
   importieren. Ursache war der globale `.gitignore`-Eintrag `data/`.

Das Rezept installiert deshalb einen vollständigen, gepinnten Checkout des
offiziellen SAM3-Repositories in das Image. Die zugriffsgeschützten Gewichte
werden nach dem Build separat über dein Hugging-Face-Konto heruntergeladen.

## 1. Dateien nach `~/singularity` kopieren

Auf login3:

```bash
mkdir -p ~/singularity
cp ~/Foundation-Open-Voc-Segmentation-Models/singularity/sam3* ~/singularity/
cd ~/singularity
```

Falls dein Projektordner anders heißt, wird später `SAM3_PROJECT_ROOT` gesetzt.

## 2. Image bauen

```bash
cd ~/singularity
singularity build --fakeroot sam3_master.simg sam3.recipe
```

Das Rezept enthält:

- Ubuntu 24.04 und CUDA 12.8.1
- Python 3.12 in `/opt/sam3_env`
- PyTorch 2.10.0 und Torchvision 0.25.0 mit CUDA 12.8
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

Sobald dein Antrag für `facebook/sam3` genehmigt wurde:

```bash
cd ~/singularity
singularity exec sam3_master.simg hf auth login
bash sam3_download_checkpoint.sh
```

Danach müssen diese Dateien existieren:

```bash
ls -lh ~/singularity/models/sam3.pt
ls -lh ~/singularity/models/config.json
```

Der Checkpoint wird nicht in das Image eingebaut. Dadurch bleibt dein
Hugging-Face-Token außerhalb des Images und der Checkpoint kann unabhängig vom
Container ausgetauscht werden.

## 4. Smoke-Test

```bash
cd ~/singularity
sbatch sam3_smoke.sbatch \
  --image /home/eker/pfad/testbild.png \
  --prompt "lungs"
```

Der Job verwendet die kostenlose `test`-Partition, eine GPU und FP16. Logs
werden als `sam3_smoke.JOBID.out` und `sam3_smoke.JOBID.err` gespeichert.

Wenn das Projekt auf dem Cluster anders heißt:

```bash
export SAM3_PROJECT_ROOT=/home/eker/PFAD/ZUM/PROJEKT
sbatch sam3_smoke.sbatch --image /home/eker/pfad/testbild.png --prompt "lungs"
```

## 5. NIH-Textbenchmark

Gemäß TCML-Empfehlung wird der Bilddatensatz für den Job nach
`/scratch/$SLURM_JOB_ID` kopiert:

```bash
cd ~/singularity
export SAM3_PROJECT_ROOT=/home/eker/Foundation-Open-Voc-Segmentation-Models
export SAM3_STAGE_DATASET=/home/eker/datasets/nih-images

sbatch sam3_nih.sbatch \
  --bbox-csv "$SAM3_PROJECT_ROOT/Foundation & Open-Vocabulary Segmentation Models/BBox_List_2017.csv" \
  --label Atelectasis \
  --text-prompt "atelectasis" \
  --max-annotations 50
```

Der Benchmark fordert eine A4000 in der `day`-Partition an und verwendet BF16.
Ergebnisse werden unter
`$SAM3_PROJECT_ROOT/sam3_outputs/nih_atelectasis` gespeichert.

Ohne Staging kann stattdessen direkt ein Bildpfad übergeben werden:

```bash
unset SAM3_STAGE_DATASET
sbatch sam3_nih.sbatch \
  --image-root /common/ODER/HOME/PFAD/nih-images \
  --bbox-csv /home/eker/PFAD/BBox_List_2017.csv
```
