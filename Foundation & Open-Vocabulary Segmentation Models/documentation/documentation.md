# Benchmarking Foundation & Open-Vocabulary Segmentation Models on Chest X-rays

## 1. Project Motivation

Foundation segmentation models such as the Segment Anything Model (SAM) promise strong zero-shot segmentation capabilities across diverse visual domains. However, medical imaging differs substantially from natural image domains: disease regions in chest X-rays are often subtle, low-contrast, and semantically complex.

This project investigates how well promptable segmentation models perform on medical X-ray data. Instead of training a new medical model from scratch, we focus on evaluating existing foundation models under different prompting strategies.

The central question is:

**How does segmentation quality change when moving from strong visual prompts, such as ground-truth bounding boxes, to weaker prompts such as points or text-based open-vocabulary prompts?**

## 2. Dataset

We use the NIH Chest X-ray dataset together with the provided bounding-box annotations from `BBox_List_2017.csv`.

For the current benchmark, we focus on the pathology class:

**Atelectasis**

The bounding boxes are treated as weak ground truth annotations. Since the NIH annotations provide bounding boxes rather than precise pixel-level segmentation masks, our evaluation should be understood as a **localization benchmark**, not as a full medical pixel-mask segmentation benchmark.

## 3. Evaluated Prompting Strategies

We evaluate three prompt settings:

### 3.1 SAM with Ground-Truth Box Prompt

In this setting, SAM receives the physician-provided bounding box as input prompt.

This represents a strong visual prompt and acts as an upper-bound style baseline: the model already receives a spatially accurate region of interest and only needs to produce a segmentation mask inside that area.

### 3.2 SAM with Point Prompt

In this setting, SAM receives only the center point of the physician bounding box.

This is a weaker prompt because the model knows only one point inside the region of interest, but not the spatial extent of the pathology.

### 3.3 Text Prompt with GroundingDINO + SAM

SAM itself does not process text prompts directly. Therefore, we use a two-stage open-vocabulary pipeline:

```text
Text prompt → GroundingDINO predicts a bounding box → SAM segments the predicted box
```

For the current experiment, the text prompt is:

```text
atelectasis
```

This experiment evaluates whether an open-vocabulary text-based model can localize a medical pathology in a chest X-ray image.

## 4. Evaluation Metrics

Because the dataset provides bounding boxes rather than pixel-level masks, we evaluate the predicted SAM masks against the physician bounding boxes using localization-oriented metrics.

### 4.1 Mask Inside Ground-Truth Box

This metric measures how much of the predicted mask lies inside the physician bounding box.

High values indicate that the predicted mask stays within the annotated pathology region.

### 4.2 Ground-Truth Box Covered by Mask

This metric measures how much of the physician box is covered by the predicted mask.

High values indicate that the model covers the annotated region, but this metric alone can be misleading if the predicted mask is extremely large.

### 4.3 Predicted-Box IoU with Ground-Truth Box

We compute a bounding box around the predicted SAM mask and compare it with the physician bounding box using Intersection over Union.

This is the most important localization metric in our current benchmark.

### 4.4 Mask Area

The total predicted mask area is used to detect oversegmentation. Very large masks indicate that the model may segment broad anatomical regions instead of the actual pathology.

## 5. Experimental Setup

We evaluated 50 chest X-ray images from the Atelectasis subset.

All experiments were run on CPU. The evaluated pipelines were:

| Experiment   | Model/Pipeline      | Prompt                        |
| ------------ | ------------------- | ----------------------------- |
| Box Prompt   | SAM                 | Physician bounding box        |
| Point Prompt | SAM                 | Center point of physician box |
| Text Prompt  | GroundingDINO + SAM | "atelectasis"                 |

## 6. Results

### 6.1 Mean Results over 50 Atelectasis Images

| Prompt Type                | SAM Score | Mask Inside GT Box | GT Box Covered by Mask | Pred-Box IoU with GT Box | Mean Mask Area |
| -------------------------- | --------: | -----------------: | ---------------------: | -----------------------: | -------------: |
| SAM Box Prompt             |    0.8628 |             0.9499 |                 0.6056 |                   0.7780 |         23,928 |
| SAM Point Prompt           |    0.9286 |             0.0589 |                 0.9741 |                   0.0495 |        787,467 |
| Text → GroundingDINO → SAM |    0.9891 |             0.0383 |                 0.9800 |                   0.0376 |        983,474 |

### 6.2 Median Results over 50 Atelectasis Images

| Prompt Type                | SAM Score | Mask Inside GT Box | GT Box Covered by Mask | Pred-Box IoU with GT Box | Median Mask Area |
| -------------------------- | --------: | -----------------: | ---------------------: | -----------------------: | ---------------: |
| SAM Box Prompt             |    0.8647 |             0.9664 |                 0.6080 |                   0.7963 |           19,902 |
| SAM Point Prompt           |    0.9418 |             0.0503 |                 1.0000 |                   0.0422 |          943,477 |
| Text → GroundingDINO → SAM |    0.9898 |             0.0345 |                 1.0000 |                   0.0336 |        1,015,280 |

## 7. Interpretation

The results show a clear dependency on prompt type.

### Box prompts produce the best localization quality

The box-prompt setting achieves the highest predicted-box IoU with the physician annotation. The predicted masks remain mostly inside the ground-truth box, indicating that SAM can generate plausible local segmentations when given strong spatial guidance.

### Point prompts lead to severe oversegmentation

The point-prompt setting performs much worse. Although the predicted masks often cover the physician box, they are extremely large. This suggests that SAM frequently segments large anatomical structures, such as lung or thorax regions, rather than the subtle pathology itself.

### Text prompts fail mainly at the grounding stage

The open-vocabulary text pipeline performs worst in terms of localization. GroundingDINO usually predicts a box, but this box has very low overlap with the physician annotation. SAM then segments the incorrectly grounded region. Therefore, the main failure is not necessarily SAM's mask generation, but the text-to-box grounding step.

### SAM's internal score is not medically reliable

Interestingly, the internal SAM score is highest for the text-prompt pipeline, even though this setting has the worst localization performance. This shows that SAM's internal confidence score does not reliably indicate medical correctness.

## 8. Current Conclusion

Our preliminary benchmark shows that SAM can be useful for medical X-ray segmentation only when strong visual prompts are available. With weaker prompts, especially text-based open-vocabulary prompts, performance drops significantly.

This supports the hypothesis that general-purpose foundation models have strong geometric segmentation capabilities but still struggle with semantic understanding in medical imaging domains.

## 9. Limitations

This benchmark currently uses bounding-box annotations rather than pixel-level segmentation masks. Therefore, the results measure localization quality and oversegmentation behavior, not true Dice-based medical segmentation accuracy.

The current experiment is limited to 50 Atelectasis examples. Future experiments should include more pathology classes and, if possible, datasets with pixel-level medical segmentation masks.

## 10. Next Steps

Possible next steps are:

1. Extend the benchmark to all available pathology classes in the bounding-box CSV.
2. Test additional text prompt variants, such as "collapsed lung region" or "atelectasis in chest x-ray".
3. Compare SAM with a medical adaptation such as MedSAM.
4. Add a dataset with pixel-level masks to compute Dice and pixel-IoU.
5. Perform a qualitative failure analysis using representative example images.
