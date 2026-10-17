# Tomato Disease Diagnosis System — Architecture Specification

## 1. Project summary

A software-only system for diagnosing tomato plant disease, covering two visual domains with **two different model types** (not a uniform classification system — see note below):

- **Leaf disease classification** — source: PlantVillage dataset (Kaggle, `emmarex/plantdisease`), filtered to tomato-only classes. 10 classes: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, Healthy. Plain image classification.
- **Fruit disease detection** — source: `profnourasemary/tomato-dataset` (Kaggle), exported via Roboflow as "Tomato Fruit Diseases Detection v3," dated 2025-04-24. This is a **YOLOv8-format object detection dataset** (bounding box annotations per image, not folder-per-class images), 1319 images total.
  - **Augmentation already baked in**: Roboflow generated 3 versions of most source images. Audited count: 549 unique source groups (385 triplets + 164 singletons), not 1319 — see `data/AUDIT.md`. This has a direct consequence for splitting — see section 2.8.
  - Local setup: see `data/README.md` (`data/fruit_raw/`)

Leaf data setup: see `data/README.md` (`data/leaf_raw/PlantVillage/`)

**Why two model types instead of one:** the fruit dataset arrived pre-annotated with bounding boxes. Collapsing that to single-label classification would discard defect location and size. The fruit branch uses YOLOv8 detection throughout, including box-geometry severity scoring.

No physical hardware exists yet. Phase 3 (sensor telemetry + fuzzy-logic dosing + actuator control) is **explicitly out of scope** for this build; the output JSON schema is stable for future integration.

## 2. Design principles (read before writing code)

1. **Two domains, two different model types, one router.** Leaf is image classification (10 classes). Fruit is object detection (YOLOv8, 8 classes). Router: MobileNetV3-Small binary classifier.
2. **Domain-specific masking happens after routing, not before.**
3. **Severity scoring differs by branch.** Leaf: HSV/Lab segmentation on canopy. Fruit: box geometry (v1 heuristic — see fruit test report).
4. **Transfer learning, not training from scratch.** RTX 3050 Ti class GPU (~4GB VRAM). Leaf 224×224; YOLO 640×640.
5. **Class imbalance handled explicitly (leaf branch).** Weighted loss; macro-F1 and per-class metrics.
6. **INT8 quantization is deployment-time only** (deferred).
7. **Config-driven training.** Router + leaf share `train.py`; fruit uses Ultralytics via `train_fruit_yolo.py`.
8. **Split integrity for fruit:** group by Roboflow source identity before splitting (`split_fruit_groups.py`).

## 3. Pipeline (phase 1 + phase 2 only)

```
Input image
   → Preprocessing (resize, normalize, CLAHE)
   → Router: leaf vs fruit (MobileNetV3-Small)
        ├─ leaf  → Leaf classifier (EfficientNet-B0, 10 classes)
        │            → Grad-CAM overlay + HSV/Lab severity_pct
        └─ fruit → Fruit detector (YOLOv8n, 8 classes)
                     → bounding boxes + box-geometry severity_pct
   → Unified JSON output schema
```

Phase 3 (deferred): sensors → fuzzy logic → actuators. Not implemented.

## 4. Unified output schema

```json
{
  "domain": "leaf | fruit",
  "class": "string",
  "confidence": 0.0,
  "severity_pct": 0.0,
  "gradcam_overlay": "path or null",
  "detections": "array or null",
  "model_version": "string"
}
```

These seven keys are the stable contract. Later versions only **add** keys
(currently `router_confidence`, `warnings`); existing keys are never
renamed, removed or given new meanings.

## 5. Repository structure

See project root `README.md` for the current tree.

## 6. Build order

1. Data audit (`data/AUDIT.md`)
2. `split_fruit_groups.py`
3. `datasets.py` + `transforms.py`
4. Train router
5. Train leaf classifier
6. Train fruit YOLO
7. Grad-CAM
8. Severity scoring
9. `infer_pipeline.py`
10. Held-out evaluation reports
11. INT8 export (deferred)

## 7. Explicitly out of scope

GPIO/ADC sensors, fuzzy logic, actuators, camera drivers, edge display rendering.
