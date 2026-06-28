from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def require_complete_sam3_install() -> Path:
    """Fail early when Python resolves to the incomplete vendored SAM3 copy."""
    spec = importlib.util.find_spec("sam3")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "SAM3 ist nicht installiert. Verwende das Singularity-Image aus "
            "scripts/build_sam3_container.sh."
        )

    package_dir = Path(spec.origin).resolve().parent
    required_file = package_dir / "train" / "data" / "collator.py"
    if not required_file.is_file():
        raise RuntimeError(
            "Die gefundene SAM3-Installation ist unvollständig: "
            f"{required_file} fehlt. Installiere den offiziellen Checkout über "
            "das Singularity-Rezept singularity/SAM3.def."
        )
    return package_dir


def require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA ist nicht verfügbar. Starte das Programm innerhalb eines GPU-Jobs "
            "und prüfe die CUDA-PyTorch-Installation."
        )
    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        raise RuntimeError(
            f"Die zugewiesene GPU {torch.cuda.get_device_name()} hat Compute "
            f"Capability {capability[0]}.{capability[1]}. Der installierte "
            "PyTorch-2.10-/CUDA-12.8-Build benötigt mindestens 7.0. Fordere "
            "auf TCML eine A4000 an."
        )
    return "cuda"


def load_image_model(checkpoint: str | None, confidence_threshold: float):
    require_complete_sam3_install()
    device = require_cuda()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint,
        load_from_HF=checkpoint is None,
        eval_mode=True,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    return model, processor, device


@contextmanager
def cuda_inference_context(precision: str):
    import torch

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None,
    }
    if precision not in dtypes:
        raise ValueError(f"Unbekannte Präzision: {precision}")

    with torch.inference_mode():
        if dtypes[precision] is None:
            yield
        else:
            with torch.autocast(device_type="cuda", dtype=dtypes[precision]):
                yield


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    # NumPy cannot convert PyTorch bfloat16 tensors directly.
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def select_highest_score(output: dict[str, Any]):
    scores = tensor_to_numpy(output["scores"]).reshape(-1)
    if scores.size == 0:
        return None

    best_index = int(np.argmax(scores))
    mask = tensor_to_numpy(output["masks"][best_index]).squeeze().astype(bool)
    box = tensor_to_numpy(output["boxes"][best_index]).reshape(-1).tolist()
    return {
        "index": best_index,
        "score": float(scores[best_index]),
        "mask": mask,
        "box": [float(v) for v in box],
        "num_detections": int(scores.size),
    }


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def mask_to_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None
    # x2/y2 are exclusive, matching area computations and array slicing.
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def evaluate_mask_against_box(mask: np.ndarray, gt_box: list[float]) -> dict[str, float]:
    mask = mask.astype(bool)
    height, width = mask.shape
    x1, y1, x2, y2 = gt_box
    x1 = max(0, min(width, round(x1)))
    y1 = max(0, min(height, round(y1)))
    x2 = max(0, min(width, round(x2)))
    y2 = max(0, min(height, round(y2)))

    gt_region = np.zeros_like(mask, dtype=bool)
    gt_region[y1:y2, x1:x2] = True
    intersection = int(np.logical_and(mask, gt_region).sum())
    mask_area = int(mask.sum())
    gt_area = int(gt_region.sum())
    predicted_box = mask_to_box(mask)

    return {
        "mask_area": float(mask_area),
        "mask_inside_gt_box": intersection / mask_area if mask_area else 0.0,
        "gt_box_covered_by_mask": intersection / gt_area if gt_area else 0.0,
        "pred_box_iou_with_gt_box": (
            box_iou(predicted_box, gt_box) if predicted_box is not None else 0.0
        ),
    }
