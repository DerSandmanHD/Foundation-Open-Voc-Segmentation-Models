# SAM3 COCO box-prompt experiment — venv runner

This version uses an existing Python virtual environment named `seg-project`.
The runner does not create an environment or install packages.

## Expected layout

Run the shell script from a directory containing:

```text
seg-project/bin/activate
```

The COCO dataset should have this structure:

```text
/data/coco/
  annotations/instances_val2017.json
  val2017/*.jpg
```

## Run

```bash
chmod +x run_experiment_venv.sh
COCO_ROOT=/data/coco ./run_experiment_venv.sh
```

Internally, the script activates the environment exactly as follows:

```bash
source seg-project/bin/activate
```

Example with experiment overrides:

```bash
COCO_ROOT=/data/coco \
NUM_IMAGES=100 \
NUM_CLASSES=10 \
NUM_PERTURBATIONS=5 \
MAX_EDGE_SHIFT=0.05 \
SEED=123 \
OUTPUT=./results/run_seed123.jsonl \
./run_experiment_venv.sh
```

The Python inference program remains `run_sam3_coco_box_robustness.py`.
