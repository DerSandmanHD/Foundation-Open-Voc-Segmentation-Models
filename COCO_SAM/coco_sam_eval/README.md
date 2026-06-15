# SAM COCO evaluation

This folder contains a small script to test Segment Anything on a COCO subset with two prompt types:

- bounding box prompts
- point prompts

It also writes a CSV with evaluation metrics and a few example overlays.

## Expected inputs

- COCO image folder, for example `val2017`
- COCO annotations file, for example `instances_val2017.json`
- a SAM checkpoint file, for example `sam_vit_b_01ec64.pth`
- the local cloned SAM repository, for example `C:\Master\ML Praktikum\SAM Repo\segment-anything`

## Example command

```powershell
python .\coco_sam_eval\evaluate_sam_coco.py `
  --coco_images "D:\coco\val2017" `
  --coco_annotations "D:\coco\annotations\instances_val2017.json" `
  --checkpoint "D:\models\sam_vit_b_01ec64.pth" `
  --sam_repo_root "C:\Master\ML Praktikum\SAM Repo\segment-anything" `
  --model_type vit_b `
  --max_annotations 50 `
  --save_examples 10
```

## Output

The script writes these files into `results/`:

- `sam_coco_results.csv`
- `sam_coco_summary_mean.csv`
- `sam_coco_summary_median.csv`
- `examples/` with visual overlays
