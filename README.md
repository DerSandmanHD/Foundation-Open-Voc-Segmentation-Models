# Benchmarking Foundation & Open-Vocabulary Segmentation Models on Chest X-rays

This repository contains an evaluation pipeline designed to assess the zero-shot generalization capabilities of general-purpose foundation segmentation models on medical imaging, specifically Chest X-rays (CXRs). 

The core of this project investigates how segmentation and localization quality change when moving from strong visual prompts (Ground-Truth annotations) to weaker prompts (points) and text-based open-vocabulary configurations.

---

## 1. Project Motivation & Objectives

Foundation models like the Segment Anything Model (SAM) have introduced a powerful paradigm: segmenting objects from text or visual prompts without task-specific training. However, medical imaging differs substantially from natural image domains: disease regions in chest X-rays are often subtle, low-contrast, and semantically complex.

This project addresses key open research questions:
* **Domain Generalization:** Can foundation models truly generalize to unseen, safety-critical medical domains?
* **Prompt Sensitivity:** How heavily does the model depend on the strength of the input prompt?
* **Open-Vocabulary Viability:** Can text-to-box grounding models accurately locate medical concepts using natural language alone?

Since chest X-ray datasets frequently provide bounding boxes rather than precise pixel-level masks, this framework functions as a localization and oversegmentation benchmark to identify key failure modes (e.g., domain gaps or structural drift).

---

## 2. Evaluated Prompting Strategies

The framework benchmarks three distinct prompt modalities to map out the performance boundaries of zero-shot medical localization:

### 2.1 Direct Ground-Truth Box Prompt
* **Mechanism:** Passes the precise, human-annotated expert bounding box coordinates directly into the foundation model.
* **Purpose:** Acts as an upper-bound baseline to evaluate the model's pure geometric edge-snapping and completion capabilities when strict spatial limits are predefined.

### 2.2 Weak Point Prompt
* **Mechanism:** Derives the mathematical center point of the expert region and feeds it as a single coordinate point prompt.
* **Purpose:** Evaluates how the model performs in an interactive segmentation scenario where it knows where a pathology is, but lacks information about its spatial boundaries.

### 2.3 Open-Vocabulary Text Prompt Cascading
* **Mechanism:** A two-stage pipeline designed for fully autonomous text-to-mask localization.
  
  Text Prompt (Pathology Name) -> Grounding Detector (Predicts Box) -> Segmentation Model (Generates Mask)

* **Purpose:** Evaluates whether a generic text-conditioned detector can successfully resolve medical terminology on an X-ray image to seed the segmentation model correctly.

---

## 3. System Mechanics & Core Insights

The evaluation metrics focus on tracking structural behaviors rather than standard pixel-perfect coefficients:
* **Localization Accuracy:** Comparing bounding boxes fitted around generated masks against expert boxes using Intersection over Union (IoU).
* **Oversegmentation Tendency:** Monitoring the total predicted mask area to detect when the model fails to isolate a subtle pathology and instead segments massive anatomical features (like an entire lung field or the whole thorax).
* **Grounding vs. Segmentation Performance:** Isolating whether a failure in the text pipeline is caused by poor mask generation or an inaccurate initial text-to-box translation.
* **Confidence Verification:** Analyzing whether the model's internal confidence/IoU score correlates with actual medical accuracy.

---

## 4. Core Workspace Components

* **Dataset Anchors:** Pathologies filtered from medical bounding box datasets (e.g., the NIH Chest X-ray dataset).
* **Geometric Execution Suite:** Codebase handling direct spatial input data, point mapping, and foundation model tracking.
* **Open-Vocabulary Suite:** Script configurations connecting text encoders, zero-shot detectors, and segmentation decoders.
* **Analytics Tooling:** Scripts to generate statistical metric distributions (barplots and boxplots) to visually compare the stability of each prompt type.
