# Project Report — Version 1
## Tomato Disease Diagnostic System (Four-Class Classifier)

**Branch:** `4-class-classifier`  
**Date:** 2026-07-07  
**Written for:** Future developers joining this project

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [Why it was built this way](#2-why-it-was-built-this-way)
3. [Architecture and system design](#3-architecture-and-system-design)
4. [Repository structure — every file explained](#4-repository-structure--every-file-explained)
5. [The datasets](#5-the-datasets)
6. [Data preprocessing](#6-data-preprocessing)
7. [The three models](#7-the-three-models)
8. [How training works](#8-how-training-works)
9. [How inference works](#9-how-inference-works)
10. [The four-class system (this branch)](#10-the-four-class-system-this-branch)
11. [The Gradio web interface](#11-the-gradio-web-interface)
12. [Test results — all numbers explained](#12-test-results--all-numbers-explained)
13. [How to set up and run from scratch](#13-how-to-set-up-and-run-from-scratch)
14. [Known limitations](#14-known-limitations)
15. [Glossary](#15-glossary)

---

## 1. What this project does

This system takes a photo of a tomato plant and automatically tells you what condition it is in.

**On this branch (`4-class-classifier`), it classifies any tomato image into exactly one of four conditions:**

| Output | What it means |
|---|---|
| **Early blight** | The leaf has a fungal disease called Early blight (Alternaria solani). Brown target-ring lesions on the leaf. |
| **Blossom_End_Rot** | The fruit has a calcium-deficiency disorder. A dark sunken patch appears at the bottom of the tomato. |
| **Healthy leaf** | The leaf looks normal — no disease detected. |
| **Healthy plant** | The fruit looks normal — no disorder detected. |

The system also tells you **how confident** it is (0–100%), **which part of the image it was looking at** (Grad-CAM heatmap for leaves, bounding boxes for fruit), and automatically saves a result image and a log file every time you test an image.

---

## 2. Why it was built this way

### The core design challenge

Tomato disease comes in two completely different visual forms:
- **Leaf diseases** look like colour changes or patterns on a flat leaf surface → best handled by an **image classifier**
- **Fruit disorders** involve specific regions/shapes on a 3D fruit → best handled by an **object detector** that draws boxes around the defect

A single model architecture cannot handle both optimally. This is why the system uses **three separate models**:

1. **Router** — first decides: "is this a leaf or a fruit photo?"
2. **Leaf classifier** — handles leaf photos (10 possible leaf disease classes)
3. **Fruit detector** — handles fruit photos (8 possible fruit disorder classes)

### Why not retrain a new single model for 4 classes?

The existing models were already trained and tested on large datasets with high accuracy. Building the 4-class system by **extracting scores from the existing models** means:
- Zero new training time
- The 4-class system inherits all the performance of the original models
- Any future improvement to the base models automatically improves the 4-class system too

---

## 3. Architecture and system design

---

### 3.1 High-level system overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                  TOMATO DISEASE DIAGNOSTIC SYSTEM                    ║
║                      (4-class-classifier branch)                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ┌──────────────┐      ┌─────────────────────────────────────────┐ ║
║   │   User /     │      │             INFERENCE LAYER              │ ║
║   │   Web UI     │─────▶│                                         │ ║
║   │  (Gradio)    │      │  ┌──────────┐   leaf  ┌──────────────┐  │ ║
║   └──────────────┘      │  │  Router  │────────▶│  Leaf Model  │  │ ║
║                         │  │  (MNv3)  │         │  (EffNet-B0) │  │ ║
║   ┌──────────────┐      │  │          │  fruit  └──────┬───────┘  │ ║
║   │  CLI / API   │─────▶│  │ leaf vs  │         ┌──────▼───────┐  │ ║
║   │  (Python)    │      │  │  fruit   │────────▶│ Fruit Model  │  │ ║
║   └──────────────┘      │  └──────────┘         │  (YOLOv8n)   │  │ ║
║                         │                       └──────┬───────┘  │ ║
║                         │                              │           │ ║
║                         │         ┌────────────────────▼─────────┐ │ ║
║                         │         │   4-Class Score Extractor    │ │ ║
║                         │         │  Early blight / BER /        │ │ ║
║                         │         │  Healthy leaf / Healthy plant │ │ ║
║                         │         └────────────────────┬─────────┘ │ ║
║                         └──────────────────────────────┼───────────┘ ║
║                                                        │             ║
║   ┌────────────────────────────────────────────────────▼───────────┐ ║
║   │                    OUTPUT LAYER                                 │ ║
║   │  predicted_class · confidence · class_scores · Grad-CAM ·      │ ║
║   │  YOLO boxes · domain · saved result images · JSON log          │ ║
║   └─────────────────────────────────────────────────────────────── ┘ ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

### 3.2 Full inference data flow

This diagram shows exactly what happens to an image from the moment it enters the system to the final output:

```
                          INPUT IMAGE
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Image Preprocessing   │
                 │  • Open with PIL        │
                 │  • Convert to RGB       │
                 │  • CLAHE equalisation   │
                 │  • Resize → 224×224     │
                 │  • ToTensor             │
                 │  • Normalise (ImageNet) │
                 └────────────┬───────────┘
                              │  tensor [1, 3, 224, 224]
                              ▼
                 ┌────────────────────────┐
                 │      ROUTER MODEL       │
                 │  MobileNetV3-Small      │
                 │  Input:  224×224 tensor │
                 │  Output: [p_leaf,       │
                 │           p_fruit]      │
                 │  Argmax → 0 or 1        │
                 └──────┬─────────┬───────┘
                        │         │
                   p_leaf>p_fruit  p_fruit>p_leaf
                        │         │
           ┌────────────▼─┐     ┌─▼────────────────┐
           │  LEAF BRANCH  │     │   FRUIT BRANCH    │
           │               │     │                   │
           │ EfficientNet  │     │    YOLOv8n        │
           │    B0         │     │                   │
           │               │     │ Input: image path │
           │ Input: tensor │     │ Output: boxes     │
           │ Output:       │     │  [x1,y1,x2,y2,   │
           │  10 softmax   │     │   conf, class_id] │
           │  probs        │     │                   │
           │               │     │ For each box:     │
           │ Extract:      │     │  Extract:         │
           │  idx1 = p_eb  │     │  cls1 → BER conf  │
           │  idx2 = p_hl  │     │  cls4 → HP conf   │
           └──────┬────────┘     └────────┬──────────┘
                  │                       │
                  ▼                       ▼
           ┌────────────┐         ┌───────────────┐
           │  GRAD-CAM   │         │  YOLO BOX     │
           │  Heatmap    │         │  OVERLAY      │
           │  (explainab)│         │  (explainab)  │
           └──────┬──────┘         └──────┬────────┘
                  │                       │
                  └───────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   4-CLASS RESULT DICT   │
                 │                        │
                 │  predicted_class: str  │
                 │  confidence: float     │
                 │  domain: str           │
                 │  router_confidence: f  │
                 │  class_scores: dict    │
                 │  detections: list|None │
                 │  raw_leaf_top3: list   │
                 │  model_versions: dict  │
                 └────────────┬───────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
           ┌──────────────┐    ┌──────────────────┐
           │  Gradio UI   │    │  File System      │
           │  (browser)   │    │                   │
           │              │    │ *_result.jpg      │
           │  3 columns:  │    │ *_explain.jpg     │
           │  • upload    │    │ inference_log.json│
           │  • heatmap/  │    └──────────────────┘
           │    boxes     │
           │  • scores    │
           └──────────────┘
```

---

### 3.3 Training pipeline design

The training pipeline is separate from inference and was run once to produce the three `.pt` checkpoint files.

```
╔══════════════════════════════════════════════════════════╗
║                   TRAINING PIPELINE                       ║
╚══════════════════════════════════════════════════════════╝

  RAW DATA SOURCES
  ┌──────────────────┐    ┌──────────────────────────┐
  │  PlantVillage    │    │  Roboflow Fruit Dataset   │
  │  (16k images,   │    │  (1319 images + YOLO .txt │
  │   10 folders)   │    │   labels, 440 unique src) │
  └────────┬─────────┘    └────────────┬─────────────┘
           │                           │
           ▼                           ▼
  ┌──────────────────┐    ┌──────────────────────────┐
  │ split_leaf_      │    │ split_fruit_groups.py     │
  │ manifest.py      │    │                           │
  │                  │    │ Groups images by Roboflow │
  │ Stratified split │    │ source stem BEFORE split  │
  │ 70/15/15, seed42 │    │ → prevents augmentation  │
  │                  │    │   leakage across splits  │
  │ → train.csv      │    │ → images/train|val|test/  │
  │   val.csv        │    │   labels/train|val|test/  │
  │   test.csv       │    │   data.yaml               │
  │   class_map.json │    └────────────┬─────────────┘
  └────────┬─────────┘                 │
           │                           │
           ▼                           │
  ┌──────────────────┐                 │
  │ build_router_    │                 │
  │ manifest.py      │                 │
  │                  │                 │
  │ Combines leaf    │                 │
  │ + fruit images   │                 │
  │ with domain label│                 │
  │ → router/        │                 │
  │   train|val|test │                 │
  └────────┬─────────┘                 │
           │                           │
  ─────────┼───────────────────────────┼──────────────────
           ▼                           ▼
  ┌────────────────────┐   ┌───────────────────────────┐
  │   train.py         │   │  train_fruit_yolo.py       │
  │  (router + leaf)   │   │  (Ultralytics YOLO)        │
  │                    │   │                            │
  │  Phase 1:          │   │  Pre-trained: yolov8n.pt   │
  │  • Freeze backbone │   │  (COCO weights)            │
  │  • Train head only │   │                            │
  │  • 10 epochs max   │   │  Fine-tune on fruit split: │
  │  • Monitor val F1  │   │  • 100 epochs max          │
  │                    │   │  • patience=20             │
  │  Phase 2:          │   │  • 640×640 input           │
  │  • Unfreeze last   │   │  • batch=8                 │
  │    3 feat blocks   │   │                            │
  │  • LR = 0.00005    │   │  Best weights saved to:    │
  │  • 15 epochs max   │   │  runs/fruit/train/         │
  │  • Early stop p=4  │   │    weights/best.pt         │
  └────────┬───────────┘   └──────────────┬────────────┘
           │                              │
           ▼                              ▼
  ┌──────────────────────────────────────────────────────┐
  │                  checkpoints/                         │
  │                                                      │
  │  router_v1_mobilenet_v3_small.pt   (2 classes)       │
  │  leaf_v1_efficientnet_b0.pt        (10 classes)      │
  │  fruit_v1_yolov8n.pt               (8 classes)       │
  └──────────────────────────────────────────────────────┘
```

---

### 3.4 Model architecture details

#### Router — MobileNetV3-Small

```
Input image 224×224×3
        │
        ▼
┌───────────────────────────────────────────┐
│          MobileNetV3-Small backbone        │
│  (Pre-trained on ImageNet — frozen in P1) │
│                                           │
│  Inverted residual blocks with            │
│  Squeeze-and-Excitation attention         │
│  (efficient depthwise separable convs)    │
│                                           │
│  Output feature map: 576 channels         │
└────────────────────┬──────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   AdaptiveAvgPool2d   │  → [576]
         │   Hardswish           │
         │   Linear(576 → 1024)  │
         │   Hardswish           │
         │   Dropout(0.2)        │
         │   Linear(1024 → 2)    │  ← replaced head
         └───────────┬───────────┘
                     │
                     ▼
              Softmax → [p_leaf, p_fruit]
              Argmax  → 0 (leaf) or 1 (fruit)
```

#### Leaf Classifier — EfficientNet-B0

```
Input image 224×224×3
        │
        ▼  CLAHE → resize → normalize → tensor
        │
        ▼
┌───────────────────────────────────────────┐
│         EfficientNet-B0 backbone           │
│  (Pre-trained on ImageNet)                │
│                                           │
│  features[0]:  Conv stem                 │
│  features[1]:  MBConv block ×1            │
│  features[2]:  MBConv block ×2            │
│  features[3]:  MBConv block ×2            │
│  features[4]:  MBConv block ×3            │
│  features[5]:  MBConv block ×3            │
│  features[6]:  MBConv block ×4   ◀ Grad-CAM target layer
│  features[7]:  MBConv block ×1            │
│  features[8]:  Conv head                 │
│                                           │
│  Training:                               │
│    Phase 1: only classifier trained      │
│    Phase 2: features[6][7][8] + classif  │
└────────────────────┬──────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   AdaptiveAvgPool2d   │  → [1280]
         │   Dropout(0.2)        │
         │   Linear(1280 → 10)   │  ← replaced head
         └───────────┬───────────┘
                     │
                     ▼
       Softmax → 10 probabilities
       [p_0, p_1, ..., p_9]
              │
              ├── p_1 = "Early blight" score  ◀ extracted here
              └── p_2 = "Healthy" score       ◀ extracted here
```

#### Fruit Detector — YOLOv8n

```
Input image (any size, auto-resized to 640×640)
        │
        ▼
┌───────────────────────────────────────────┐
│            YOLOv8n Backbone                │
│  (Pre-trained on COCO)                    │
│                                           │
│  CSP-Darknet layers (depthwise convs)     │
│  C2f blocks (cross-stage partial)         │
│  SPPF (Spatial Pyramid Pooling Fast)      │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│              YOLOv8 Neck                   │
│  FPN (Feature Pyramid Network)            │
│  PAN (Path Aggregation Network)           │
│  Merges features at multiple scales       │
│  (detects objects of different sizes)     │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│         Detection Head (3 scales)         │
│  For each anchor point outputs:           │
│  [x, y, w, h, objectness, class_probs×8] │
└────────────────────┬──────────────────────┘
                     │
                     ▼
         NMS (Non-Maximum Suppression)
         conf threshold = 0.20
         IoU threshold  = 0.70
                     │
                     ▼
       List of detections per image:
       [class_id, confidence, x1, y1, x2, y2]
              │
              ├── class 1 = "Blossom_End_Rot"  ◀ extracted here
              └── class 4 = "Healthy_Tomato"   ◀ extracted here
```

---

### 3.5 Four-class score extraction design

This is the core design of this branch. The diagram shows how the 4-class output is assembled from the existing model scores:

```
                 ┌─── LEAF BRANCH ─────────────────────────┐
                 │                                          │
                 │   EfficientNet-B0 softmax output         │
                 │   (10 values summing to 1.0)             │
                 │                                          │
                 │   Index:  0    1    2    3  ...  9        │
                 │   Class:  BS  [EB] [HL]  LB ...  TMV     │
                 │   Value: 0.01 0.94 0.02 0.01 ...  0.0    │
                 │               ↑    ↑                     │
                 │          extracted  extracted            │
                 │               │    │                     │
                 └───────────────┼────┼─────────────────────┘
                                 │    │
                         ┌───────┘    └────────┐
                         │                     │
                         ▼                     ▼
               ┌──────────────────┐  ┌──────────────────┐
               │  "Early blight"  │  │  "Healthy leaf"  │
               │   score = 0.94   │  │   score = 0.02   │
               └────────┬─────────┘  └────────┬─────────┘
                        │                     │
                        └──────────┬──────────┘
                                   │  max(0.94, 0.02) → 0.94
                                   ▼
                          Prediction: Early blight
                          Confidence: 0.9400


                 ┌─── FRUIT BRANCH ────────────────────────┐
                 │                                          │
                 │   YOLOv8n detection output               │
                 │   (variable number of boxes)             │
                 │                                          │
                 │   Box 1: class=1 (BER),  conf=0.69       │
                 │   Box 2: class=2 (Cat),  conf=0.26       │
                 │   Box 3: class=4 (HT),   conf=0.31       │
                 │              ↑                  ↑        │
                 │         scan for cls1      scan for cls4 │
                 │         max_conf=0.69      max_conf=0.31 │
                 └───────────────┬────────────────┬─────────┘
                                 │                │
                                 ▼                ▼
               ┌──────────────────────┐  ┌──────────────────┐
               │  "Blossom_End_Rot"   │  │  "Healthy plant" │
               │   score = 0.69       │  │   score = 0.31   │
               └──────────┬───────────┘  └────────┬─────────┘
                          │                       │
                          └──────────┬────────────┘
                                     │  max(0.69, 0.31) → 0.69
                                     ▼
                            Prediction: Blossom_End_Rot
                            Confidence: 0.6900

               Edge case: no detections at all
               → both BER=0.0, HP=0.0
               → default HP=1.0 (no pathology = healthy)
               → Prediction: Healthy plant, confidence 1.0
```

---

### 3.6 Grad-CAM architecture (explainability)

Grad-CAM explains the leaf model's decision by tracing which parts of the image caused the highest activation for the predicted class.

```
  Input image (224×224×3)
        │
        ▼
  EfficientNet-B0 forward pass
        │
        │  ┌──── Register hooks at features[-1] ────┐
        │  │  (last convolutional feature block)     │
        │  │                                         │
        ▼  ▼                                         │
  feature maps A                              gradients G
  shape: [1, C, H, W]                        shape: [1, C, H, W]
  (captured by forward hook)                 (captured by backward hook)
        │                                         │
        │                                         │ backward pass on
        │                                         │ predicted class score
        │                                         │
        └─────────────────┬───────────────────────┘
                          │
                          ▼
              weights w = mean(G, dim=[H,W])
              shape: [C, 1, 1]
                          │
                          ▼
              CAM = ReLU( sum(w × A, dim=C) )
              shape: [H, W]
                          │
                          ▼
              Resize CAM to 224×224
              Normalize to [0, 1]
                          │
                          ▼
              Apply COLORMAP_JET (blue→red)
              Blend with original image (α=0.5)
                          │
                          ▼
              Heatmap overlay image (224×224×3)
              Red = model looked here most
              Blue = model ignored this area
```

---

### 3.7 Severity scoring design

The system also estimates how severe the disease is, using two different methods depending on the domain:

```
  LEAF SEVERITY (HSV + Lab colour science)
  ─────────────────────────────────────────
  Input: RGB image
      │
      ▼  Convert to HSV
  Canopy mask = pixels where:
      Hue ∈ [25°, 95°]  (green range)
      Saturation > 40
      Value > 40
      │
      ▼  Convert to Lab colour space
  For each canopy pixel:
      deviation = |a - median_a| + |b - median_b|
      (Lab "a" and "b" measure colour shift from healthy green)
      │
      ▼  Threshold deviation > 18.0
  Lesion mask = canopy pixels with high colour deviation
      │
      ▼
  severity_pct = (lesion pixels / canopy pixels) × 100


  FRUIT SEVERITY (bounding box geometry)
  ─────────────────────────────────────────
  Input: list of YOLO detections + image dimensions
      │
      ▼
  Separate disease boxes from healthy boxes
      │
      ├── disease_area = sum of (w × h) for each disease box
      │
      └── estimated_fruit_area:
              ├── if healthy boxes exist → union of healthy boxes
              ├── if multiple boxes exist → union of all boxes
              └── if single disease box → full image area (fallback)
      │
      ▼
  severity_pct = min(100, disease_area / fruit_area × 100)
```

---

### 3.8 Data storage design

```
  checkpoints/          ← Model weights (binary, ~43 MB total, not in git)
  │
  ├── router_v1_mobilenet_v3_small.pt
  │     Contents: {model_state_dict, config, val_metrics, phase, epoch}
  │
  ├── leaf_v1_efficientnet_b0.pt
  │     Contents: {model_state_dict, config, class_names[10], val_metrics,
  │                phase, epoch, metrics_history, best_macro_f1,
  │                completed_phases, best_phase, best_epoch}
  │
  └── fruit_v1_yolov8n.pt
        Contents: Ultralytics YOLO standard format
                  Loaded via: YOLO(path).predict(...)

  data/splits/          ← Data manifests (text/CSV, committed to git)
  │
  ├── leaf/
  │     ├── train.csv   11,208 rows: image_path, label
  │     ├── val.csv      2,402 rows
  │     ├── test.csv     2,401 rows
  │     └── class_map.json  {class_names: [...], class_to_idx: {...}}
  │
  ├── fruit/
  │     ├── data.yaml   YOLO format: path, train, val, test, nc, names
  │     ├── labels/train/*.txt   YOLO box annotations (committed)
  │     ├── labels/val/*.txt
  │     └── labels/test/*.txt
  │
  └── router/
        ├── train.csv   12,118 rows: image_path, domain, domain_idx
        ├── val.csv      2,601 rows
        ├── test.csv     2,611 rows
        └── domain_map.json  {leaf: 0, fruit: 1}

  results/              ← Reports and output images
  │
  ├── four_class_test_report.json    ← Structured metrics
  ├── four_class_test_report.md      ← Human-readable report
  ├── leaf_classifier_test_report.json
  ├── fruit_detector_test_report.json
  └── inference_results/
        ├── inference_log.json       ← Appended every run: [{timestamp,
        │                               predicted_class, confidence,
        │                               domain, class_scores, paths}]
        ├── *_result.jpg             ← Annotated score panel image
        └── *_explain.jpg            ← Grad-CAM or detection overlay
```

---

### 3.9 Component interaction map

Shows which Python files call which other files:

```
  app.py
    ├── imports four_class_infer.py  (FourClassDiagnosticClassifier)
    └── imports gradcam.py           (compute_gradcam, overlay_gradcam)

  four_class_infer.py
    ├── imports evaluate.py          (load_leaf_checkpoint,
    │                                 load_router_checkpoint, repo_root)
    ├── imports transforms.py        (build_leaf_transforms)
    └── imports train_fruit_yolo.py  (resolve_data_yaml,
                                      write_resolved_data_yaml)

  infer_pipeline.py  [main branch pipeline — unchanged]
    ├── imports evaluate.py
    ├── imports gradcam.py
    ├── imports models/leaf_classifier.py
    ├── imports models/router.py
    ├── imports severity.py
    └── imports transforms.py

  evaluate.py
    ├── imports models/leaf_classifier.py  (build_leaf_classifier)
    └── imports models/router.py           (build_router)

  train.py
    ├── imports datasets.py          (build_leaf_dataloader,
    │                                 build_router_dataloader)
    ├── imports models/leaf_classifier.py
    ├── imports models/router.py
    └── imports transforms.py

  datasets.py
    └── imports transforms.py        (build_leaf_transforms)
```

---

### 3.10 Technology stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ | All code |
| **Deep learning** | PyTorch | ≥ 2.6.0 | Model training and inference |
| **Vision models** | torchvision | ≥ 0.21.0 | EfficientNet-B0, MobileNetV3 |
| **Object detection** | Ultralytics | ≥ 8.3.107 | YOLOv8n training + inference |
| **Image processing** | OpenCV (cv2) | ≥ 4.11.0 | CLAHE, HSV masking, Grad-CAM resize |
| **Image I/O** | Pillow (PIL) | ≥ 10.0.0 | Loading, converting, saving images |
| **Array math** | NumPy | ≥ 2.1.1 | Heatmap computation, severity |
| **Config** | PyYAML | ≥ 6.0 | Training configs |
| **Web UI** | Gradio | ≥ 4.0.0 | Browser-based demo interface |
| **Hardware** | NVIDIA RTX 3050 Ti | 4GB VRAM | Training (GPU) |
| **OS** | Windows 10/11 | — | Development environment |

---

## 4. Repository structure — every file explained

```
IIHR-object-detection-and-classification/
│
├── src/                          ← All Python code
│   ├── four_class_infer.py       ← THE MAIN MODULE for this branch
│   ├── app.py                    ← Gradio web interface
│   ├── infer_pipeline.py         ← Original full 10+8 class inference
│   ├── train.py                  ← Training script (router + leaf)
│   ├── train_fruit_yolo.py       ← Training script (fruit YOLO)
│   ├── evaluate.py               ← Evaluation utilities + checkpoint loaders
│   ├── datasets.py               ← PyTorch Dataset classes
│   ├── transforms.py             ← Image preprocessing pipeline
│   ├── gradcam.py                ← Grad-CAM explainability
│   ├── severity.py               ← Disease severity scoring
│   ├── split_leaf_manifest.py    ← Builds leaf train/val/test CSV files
│   ├── split_fruit_groups.py     ← Builds fruit YOLO split
│   ├── build_router_manifest.py  ← Builds router train/val/test CSV files
│   └── models/
│       ├── leaf_classifier.py    ← EfficientNet-B0 definition
│       └── router.py             ← MobileNetV3-Small definition
│
├── configs/                      ← Training configuration files
│   ├── four_class.yaml           ← Config for this branch's 4-class system
│   ├── leaf.yaml                 ← Leaf classifier training config
│   ├── router.yaml               ← Router training config
│   └── fruit_yolo.yaml           ← Fruit YOLO training config
│
├── checkpoints/                  ← Trained model weight files (.pt)
│   ├── README.md                 ← Instructions for obtaining weights
│   ├── router_v1_mobilenet_v3_small.pt   (train locally — not in git)
│   ├── leaf_v1_efficientnet_b0.pt        (train locally — not in git)
│   └── fruit_v1_yolov8n.pt               (train locally — not in git)
│
├── data/
│   ├── README.md                 ← Data setup instructions
│   └── splits/
│       ├── leaf/                 ← Leaf train/val/test CSVs + class_map.json
│       ├── fruit/                ← Fruit YOLO labels + data.yaml
│       └── router/               ← Router train/val/test CSVs + domain_map.json
│
├── results/
│   ├── four_class_test_report.md   ← Test results for this branch
│   ├── four_class_test_report.json ← Same, machine-readable
│   ├── leaf_classifier_test_report.md
│   ├── leaf_classifier_test_report.json
│   ├── fruit_detector_test_report.md
│   ├── fruit_detector_test_report.json
│   ├── project_summary.json
│   └── inference_results/        ← Auto-saved result images from the app
│       ├── inference_log.json    ← Running log of every test run
│       └── *.jpg                 ← Annotated result images
│
├── scripts/                      ← Standalone evaluation scripts
│   ├── eval_leaf_test_report.py
│   ├── eval_fruit_test_report.py
│   ├── verify_leaf_checkpoint.py
│   ├── analyze_early_blight_confusion.py
│   ├── analyze_fruit_confusion.py
│   └── analyze_swv_box_areas.py
│
├── report_version1.md            ← THIS FILE
├── README.md                     ← Quick start guide (branch-specific)
├── ARCHITECTURE.md               ← Design decisions and rationale
└── requirements.txt              ← Python dependencies
```

---

## 5. The datasets

### Dataset 1 — Leaf images (PlantVillage)

- **Source:** Kaggle dataset `emmarex/plantdisease`, filtered to tomato-only classes
- **Type:** Image classification — one folder per disease class, images inside
- **Total images:** ~16,000 tomato leaf images
- **Classes (10 total):**

| Index | Class name | Description |
|---|---|---|
| 0 | Bacterial spot | Small water-soaked spots on leaves |
| 1 | **Early blight** | Target-ring brown lesions |
| 2 | **Healthy** | No disease |
| 3 | Late blight | Dark water-soaked patches |
| 4 | Leaf Mold | Yellow patches, grey mold underneath |
| 5 | Septoria leaf spot | Small circular spots with dark margins |
| 6 | Spider mites (Two-spotted) | Fine webbing, tiny yellow dots |
| 7 | Target Spot | Brown circular spots |
| 8 | Tomato Yellow Leaf Curl Virus | Curled yellow leaves |
| 9 | Tomato mosaic virus | Mosaic yellowing pattern |

- **Local path (not in git):** `data/leaf_raw/PlantVillage/`
- **Split:** 70% train / 15% validation / 15% test (stratified by class, seed 42)
- **Split files (in git):** `data/splits/leaf/train.csv`, `val.csv`, `test.csv`

### Dataset 2 — Fruit images (Roboflow/Kaggle)

- **Source:** Kaggle `profnourasemary/tomato-dataset`, exported via Roboflow as YOLOv8 format
- **Type:** Object detection — each image has a `.txt` file with bounding box annotations
- **Total images:** 1,319 (but only ~440 unique source images — Roboflow created 3 augmented versions of each)
- **Classes (8 total):**

| ID | Class name | Description |
|---|---|---|
| 0 | Anthracnose | Dark sunken spots on ripe fruit |
| 1 | **Blossom_End_Rot** | Dark patch at the blossom end of fruit |
| 2 | Catfaced | Deformed fruit, scarring |
| 3 | Fruit_Cracking | Radial or concentric cracks |
| 4 | **Healthy_Tomato** | Normal healthy fruit |
| 5 | Late_Blight | Dark water-soaked lesions on fruit |
| 6 | Mold | Visible mold growth on fruit surface |
| 7 | Spotted_Wilt_Virus | Circular spots, bronzing |

- **Important detail about augmentation:** Because Roboflow baked augmented copies into the dataset, a naive random split could put an augmented version of a training image into the test set — which would artificially inflate test scores. This project solves that with **group-aware splitting** (see Section 5).

---

## 6. Data preprocessing

### Image preprocessing pipeline (leaf branch)

Every leaf image goes through the following steps before being fed to any model:

**Step 1 — CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Enhances local contrast in the image
- Works in LAB colour space — only applies to the L (lightness) channel
- Makes disease lesions more visible, especially in uneven lighting
- Applied using OpenCV: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`

**Step 2 — Resize to 224×224 pixels**
- EfficientNet-B0 expects 224×224 input
- Torchvision's `Resize` handles this

**Step 3 — Data augmentation (training only, not inference)**
- Random horizontal flip (50% probability)
- Random rotation up to ±15 degrees
- Colour jitter (slight brightness/contrast/saturation variation)
- This prevents the model from overfitting by artificially increasing variety

**Step 4 — Convert to tensor and normalise**
- Pixel values divided to range [0, 1], then normalized using ImageNet statistics:
  - Mean: (0.485, 0.456, 0.406)
  - Std: (0.229, 0.224, 0.225)
- This is required because EfficientNet was pre-trained on ImageNet with these values

### Fruit data split (group-aware)

**Problem:** Roboflow created 3 versions of each source image. If image `Blossom_end_root_15.jpg` has versions `...rf.abc.jpg`, `...rf.def.jpg`, `...rf.ghi.jpg`, a random split could put `abc` in training and `def` in test — the model would essentially have seen that exact image before.

**Solution (`split_fruit_groups.py`):** Group images by their Roboflow source stem (the part before `.rf.`). All augmented copies of the same source image must land in the same split. Then split at the group level 70/15/15.

---

## 7. The three models

### Model 1 — Router (MobileNetV3-Small)

**Purpose:** Binary classifier. Given any tomato image, outputs either `leaf` (0) or `fruit` (1).

**Architecture:**
- MobileNetV3-Small, pre-trained on ImageNet
- Final classifier layer replaced: original 1000-class head → 2-class head
- Input: 224×224 RGB image
- Output: 2 probabilities (leaf, fruit) — argmax is the prediction

**Why MobileNetV3-Small?**
- Very small and fast (lightweight model)
- The task is easy (leaf vs fruit is visually obvious) so a small model is enough
- Leaves more GPU memory for the heavier leaf and fruit models

**File:** `src/models/router.py`  
**Weights:** `checkpoints/router_v1_mobilenet_v3_small.pt`

---

### Model 2 — Leaf Classifier (EfficientNet-B0)

**Purpose:** Given a leaf image, classify it into one of 10 leaf disease/health classes.

**Architecture:**
- EfficientNet-B0, pre-trained on ImageNet
- Final classifier layer replaced: 1280 features → 10-class linear layer
- Input: 224×224 RGB image
- Output: 10 probabilities (one per class) via softmax

**Why EfficientNet-B0?**
- Excellent accuracy-to-size ratio — designed to be efficient
- Pre-trained on ImageNet means it already knows general visual features (edges, textures, shapes)
- Fine-tuning it (not training from scratch) is much faster on limited GPU

**Two-phase training:**
- **Phase 1:** Only the classifier head (the last layer) is trained. The backbone (feature extractor) is frozen. This prevents the pre-trained weights from being overwritten by random gradients early on.
- **Phase 2:** The last 3 feature blocks are unfrozen and trained with a very small learning rate. This allows the backbone to adapt its features to tomato leaf disease patterns.

**Class imbalance handling:**
The dataset has unequal numbers of images per class (e.g., 481 TYLCV images vs only 56 mosaic virus images). The training uses **inverse-frequency class weights** in the loss function — rarer classes are penalized more heavily when misclassified.

**File:** `src/models/leaf_classifier.py`  
**Weights:** `checkpoints/leaf_v1_efficientnet_b0.pt`

---

### Model 3 — Fruit Detector (YOLOv8n)

**Purpose:** Given a fruit image, detect and localise all visible disease/defect regions by drawing bounding boxes around them.

**Architecture:**
- YOLOv8n (nano version — smallest and fastest YOLOv8 variant)
- Pre-trained on COCO (a large general object detection dataset)
- Fine-tuned on the tomato fruit dataset
- Input: 640×640 RGB image
- Output: Bounding boxes with class labels and confidence scores

**Why object detection instead of classification?**
- The fruit dataset came with bounding box annotations (the annotators drew boxes around each disease region)
- Object detection preserves the location and size of each defect
- This enables severity scoring (how much of the fruit is diseased?)
- Multiple defects can be detected in the same image simultaneously

**File:** `src/train_fruit_yolo.py` (uses Ultralytics library)  
**Weights:** `checkpoints/fruit_v1_yolov8n.pt`

---

## 8. How training works

### Training the router and leaf classifier

Both use `src/train.py` with a YAML config file:

```
python src/train.py --config configs/router.yaml   # Train router
python src/train.py --config configs/leaf.yaml     # Train leaf classifier
```

The training loop:
1. Loads the dataset from CSV manifests via `datasets.py`
2. Applies transforms from `transforms.py`
3. Builds the model from `src/models/`
4. Runs gradient descent with early stopping on validation macro-F1
5. Saves the best checkpoint to `checkpoints/`

**Early stopping:** If the validation metric doesn't improve for 4 consecutive epochs, training stops automatically. This prevents overfitting.

**Checkpoint format** (what's inside the `.pt` file):
```python
{
    "model_state_dict":  ...,  # The actual learned weights
    "config":            ...,  # The full YAML config used
    "class_names":       [...], # Human-readable class names in order
    "val_metrics":       {...}, # Best validation metrics
    "phase":             int,   # Which training phase this checkpoint is from
    "epoch":             int,   # Which epoch
    "metrics_history":   [...], # Full training history
    "best_macro_f1":     float,
}
```

### Training the fruit detector

```
python src/train_fruit_yolo.py --config configs/fruit_yolo.yaml
```

This uses the Ultralytics library's built-in training loop. The best weights are saved to `runs/fruit/train/weights/best.pt`, which is then copied to `checkpoints/fruit_v1_yolov8n.pt`.

### Generating the data split manifests

Before training, the split CSV files must exist. If they don't:

```powershell
# Leaf splits (requires PlantVillage data in data/leaf_raw/)
python src/split_leaf_manifest.py --write

# Router splits (requires leaf splits to exist first)
python src/build_router_manifest.py --write

# Fruit splits (requires Roboflow data in data/fruit_raw/)
python src/split_fruit_groups.py
```

---

## 9. How inference works

### The original full pipeline (`src/infer_pipeline.py`)

For a given input image, the pipeline does:

```
1. Load image
2. Run router → "leaf" or "fruit"
3a. If leaf:
    - Run EfficientNet-B0 → get 10-class softmax probabilities
    - Pick the class with highest probability
    - Compute Grad-CAM heatmap (shows which region triggered the prediction)
    - Compute leaf severity % (HSV/Lab masking of diseased canopy area)
3b. If fruit:
    - Run YOLOv8n → get bounding boxes with class + confidence
    - Pick the highest-confidence detection as primary class
    - Compute fruit severity % (disease box area ÷ fruit area)
4. Return unified JSON
```

### Grad-CAM (explainability for leaves)

Grad-CAM (Gradient-weighted Class Activation Mapping) answers: **"Which pixels in the image most influenced the model's prediction?"**

How it works:
1. Run a forward pass through the leaf model (EfficientNet-B0) for the predicted class
2. Run a backward pass (compute gradients) for that predicted class's score
3. Average the gradients over the spatial dimensions → get importance weights for each feature map
4. Multiply the feature maps by their importance weights and sum → heatmap
5. Apply ReLU (only keep positive contributions)
6. Resize and overlay on the original image with a colour map (red = high importance)

If the model is working correctly, the heatmap should highlight the disease lesion area — not random background.

**File:** `src/gradcam.py`

### Severity scoring

**Leaf severity** (`src/severity.py → leaf_severity_pct()`):
1. Convert to HSV colour space
2. Create a mask of "green" pixels (the leaf canopy): hue 25–95, saturation > 40, value > 40
3. Within the canopy, compute colour deviation in Lab colour space from the median healthy colour
4. Pixels with high deviation = potentially diseased
5. `severity_pct = (diseased pixels / total canopy pixels) × 100`

**Fruit severity** (`src/severity.py → fruit_severity_pct()`):
1. Take all disease bounding boxes from YOLO
2. Sum their areas
3. Divide by estimated fruit area (using healthy tomato boxes if detected, otherwise union of all boxes)
4. `severity_pct = (disease area / fruit area) × 100`

---

## 10. The four-class system (this branch)

### What makes this branch different from `main`

The `main` branch outputs up to 18 possible classes (10 leaf + 8 fruit). This branch narrows the output to exactly **4 classes** relevant to the IIHR use case.

### How it works without new training

The key insight: the existing models **already know** the 4 target classes. We just need to extract only those scores:

```
Input image
    │
    ▼
Router → "leaf" or "fruit"
    │
    ├── leaf  → EfficientNet-B0 outputs 10 probabilities
    │              We look at only:
    │              - index 1 = "Early blight"    probability
    │              - index 2 = "Healthy"          probability
    │              Whichever is higher = prediction
    │
    └── fruit → YOLOv8n detects bounding boxes
                   We look at only:
                   - class 1 = "Blossom_End_Rot" max confidence
                   - class 4 = "Healthy_Tomato"  max confidence
                   Whichever is higher = prediction
                   (no detection at all = Healthy plant by default)
```

### Class name mapping

| Internal name (existing models) | Output name (4-class system) |
|---|---|
| "Early blight" (leaf model index 1) | **"Early blight"** |
| "Blossom_End_Rot" (YOLO class 1) | **"Blossom_End_Rot"** |
| "Healthy" (leaf model index 2) | **"Healthy leaf"** |
| "Healthy_Tomato" (YOLO class 4) | **"Healthy plant"** |

### Key file: `src/four_class_infer.py`

The `FourClassDiagnosticClassifier` class:
- `__init__()` — loads all 3 checkpoints once at startup
- `classify(image_path)` — runs the full pipeline, returns a result dict
- `class_names` — property returning `["Early blight", "Blossom_End_Rot", "Healthy leaf", "Healthy plant"]`

**Return value of `classify()`:**
```python
{
    "predicted_class":   "Early blight",   # one of the 4 classes
    "confidence":        0.9446,            # 0.0 to 1.0
    "domain":            "leaf",            # "leaf" or "fruit"
    "router_confidence": 0.9987,            # how sure the router was
    "class_scores": {                       # scores for all 4 classes
        "Early blight":    0.9446,
        "Blossom_End_Rot": 0.0,
        "Healthy leaf":    0.0218,
        "Healthy plant":   0.0,
    },
    "detections":      None,                # YOLO boxes (fruit only)
    "raw_leaf_top3":   [...],               # top 3 full leaf predictions
    "model_versions":  {...}                # model version strings
}
```

### Config file: `configs/four_class.yaml`

Documents exactly which class index from which model maps to which 4-class output. Useful reference for future developers.

---

## 11. The Gradio web interface

**File:** `src/app.py`  
**Run:** `python src/app.py`  
**Opens:** `http://127.0.0.1:7860`

### Layout (3 columns)

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│  Upload image    │  Grad-CAM / YOLO     │  Score panel         │
│                  │  detection overlay   │                      │
│  [drag & drop]   │  (explainability)    │  Predicted class     │
│                  │                      │  Confidence bar      │
│  [Classify btn]  │  Leaf: heatmap       │  4-class scores      │
│                  │  Fruit: bounding     │                      │
│  Target classes  │  boxes drawn on      │  Domain + timestamp  │
│  legend          │  original image      │                      │
└──────────────────┴──────────────────────┴──────────────────────┘
│  Score table (markdown)                                        │
│  Saved file paths                                              │
└────────────────────────────────────────────────────────────────┘
```

### Auto-saved files (every run)

| File | Contents |
|---|---|
| `results/inference_results/YYYYMMDD_HHMMSS_<class>_result.jpg` | Score panel image |
| `results/inference_results/YYYYMMDD_HHMMSS_<class>_explain.jpg` | Grad-CAM or detection overlay |
| `results/inference_results/inference_log.json` | All runs logged as JSON array |

### The explainability column

**For leaf images (Grad-CAM):**
- Red/yellow regions = where the model looked most
- If working correctly, the hot spot should be on the disease lesion
- Uses `gradcam.py → compute_gradcam()` + `overlay_gradcam_on_image()`

**For fruit images (YOLO boxes):**
- Coloured rectangles drawn on the original image for every detected defect
- Thicker boxes for the 4-class targets (Blossom_End_Rot, Healthy_Tomato)
- Thin boxes for other detected classes (informational only)
- Label shows class name + confidence score

---

## 12. Test results — all numbers explained

All testing was done on **held-out test data** — images that were never used during training or for any hyperparameter decision.

### Router

| Split | Accuracy |
|---|---:|
| Validation | 100.00% |
| Test | 100.00% |

The router never makes a routing mistake. Every leaf image goes to the leaf branch and every fruit image goes to the fruit branch.

### Leaf classifier (EfficientNet-B0) — four-class relevant classes

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Early blight | 1.0000 | 0.9800 | 0.9899 | 150 |
| Healthy leaf | 1.0000 | 1.0000 | 1.0000 | 238 |

**What these numbers mean:**
- **Precision 1.0 for Early blight** — every time the model said "Early blight", it was actually Early blight. Zero false alarms.
- **Recall 0.98 for Early blight** — the model correctly identified 147 out of 150 actual Early blight images. It missed 3 (2 → predicted Late blight, 1 → predicted Septoria).
- **F1 = 0.9899** — harmonic mean of precision and recall. Very high.
- **Healthy leaf F1 = 1.0** — perfect on the test set. All 238 healthy leaf images were correctly identified.

**Historical improvement of Early blight precision:**

| Stage | Precision | Late blight → Early blight errors |
|---|---:|---:|
| Phase 1 baseline | 0.7444 | 19 errors |
| Phase 2 val (best) | 0.9804 | 1 error |
| **Final test** | **1.0000** | **0 errors** |

This improvement happened because Phase 2 training unfroze the backbone and let the model learn the visual difference between Early blight and Late blight.

### Fruit detector (YOLOv8n) — four-class relevant classes

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Blossom_End_Rot | 0.8621 | 0.9231 | 0.9181 | 0.5656 |
| Healthy_Tomato | 0.8240 | 0.6176 | 0.7596 | 0.5016 |

**What mAP means:** Mean Average Precision. For detection tasks, the model not only needs to predict the right class — it also needs to draw a box that overlaps the ground-truth box by at least 50% (mAP@0.5) or across multiple thresholds (mAP@0.5:0.95). It's a stricter metric than classification F1.

**Why Healthy_Tomato recall is lower (0.62):** "Healthy" is inherently harder to detect — there's no specific visual feature to point at. The model sometimes doesn't draw a box around a healthy fruit even when it's present.

### Overall 4-class summary

| Class | Test score | Metric type |
|---|---:|---|
| Early blight | **0.9899** | F1 |
| Blossom_End_Rot | **0.9181** | mAP@0.5 |
| Healthy leaf | **1.0000** | F1 |
| Healthy plant | **0.7596** | mAP@0.5 |
| **4-class macro average** | **0.9169** | mixed |
| **Router** | **1.0000** | accuracy |

---

## 13. How to set up and run from scratch

### Step 1 — Clone the branch

```powershell
git clone -b 4-class-classifier https://github.com/Gowthamjignas06/IIHR-object-detection-and-classification.git
cd IIHR-object-detection-and-classification
```

### Step 2 — Create Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU (recommended):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Step 3 — Get the model weights

The `.pt` files are large binaries (~43 MB total) and are not stored in git. You have two options:

**Option A — Train them yourself (requires datasets):**

1. Download PlantVillage tomato images from Kaggle (`emmarex/plantdisease`) and place at `data/leaf_raw/PlantVillage/`
2. Download fruit data from Kaggle (`profnourasemary/tomato-dataset`) and place at `data/fruit_raw/`
3. Generate splits:
```powershell
python src/split_leaf_manifest.py --write
python src/build_router_manifest.py --write
python src/split_fruit_groups.py
```
4. Train in order:
```powershell
python src/train.py --config configs/router.yaml
python src/train.py --config configs/leaf.yaml
python src/train_fruit_yolo.py --config configs/fruit_yolo.yaml
```

**Option B — Get weights from the original developer:**  
Ask for the three `.pt` files and place them in `checkpoints/`.

### Step 4 — Run the web interface

```powershell
python src/app.py
```

Opens `http://127.0.0.1:7860` in your browser. Upload any tomato image and click Classify.

### Step 5 — Run inference from the command line

```powershell
python src/four_class_infer.py path\to\image.jpg
```

### Step 6 — Use as a Python module

```python
from src.four_class_infer import FourClassDiagnosticClassifier

clf = FourClassDiagnosticClassifier()
result = clf.classify("path/to/image.jpg")
print(result["predicted_class"])  # e.g. "Early blight"
print(result["confidence"])       # e.g. 0.9446
```

---

## 14. Known limitations

### Current limitations

| Limitation | Details | Impact |
|---|---|---|
| Out-of-scope images | If you give the system an image of a tomato with Bacterial spot (not in the 4 classes), it will still output one of the 4 classes — it cannot say "I don't know". | Low — designed for the 4 target conditions only |
| Early blight misses | 3 of 150 test Early blight images were misclassified (as Late blight or Septoria). | Low — 98% recall is high |
| Healthy plant recall | 62% recall — the YOLO model sometimes doesn't detect a healthy tomato box when the fruit is small or at the edge of the frame. | Medium — may need better fruit images |
| Fruit severity is approximate | Severity percentage for fruit is a bounding-box geometry heuristic, not a pixel-level measurement. Sensitive to how closely the photo is cropped. | Low for classification tasks |
| Image quality | Works best when the leaf or fruit fills most of the frame, good lighting, no HEIC format. | Medium — instruct users on how to photograph |
| No hardware integration | Phase 3 (sensors, actuators, fuzzy logic) is not implemented. The output JSON is designed for future integration. | Deferred by design |

### What is NOT implemented (out of scope)

- GPIO sensors (soil moisture, temperature, pH)
- Fuzzy logic dosing system
- Actuator control (irrigation, pesticide)
- Camera driver / embedded system
- INT8 quantization for edge deployment
- Mobile app

---

## 15. Glossary

| Term | Simple explanation |
|---|---|
| **EfficientNet-B0** | A type of neural network designed to be accurate and small at the same time. Used for leaf disease classification. |
| **MobileNetV3-Small** | A very lightweight neural network designed for mobile devices. Used for the leaf/fruit router. |
| **YOLOv8n** | "You Only Look Once" nano — a fast object detection model. Used for fruit disease detection. Draws boxes around objects in images. |
| **Fine-tuning** | Starting with a model already trained on a large dataset (ImageNet or COCO) and continuing training on your specific dataset. Much faster than training from scratch. |
| **Transfer learning** | The general concept behind fine-tuning — transferring knowledge from one task to another. |
| **Softmax** | A mathematical function that converts a list of scores into probabilities that sum to 1. E.g., `[2.1, 0.3, 0.1]` → `[0.85, 0.09, 0.06]`. |
| **F1 score** | A single number combining precision and recall. Ranges from 0 (worst) to 1.0 (perfect). Good for imbalanced datasets. |
| **mAP@0.5** | Mean Average Precision at IoU 0.5. A standard metric for object detection. Requires the detected box to overlap the true box by 50%+. |
| **Precision** | Of all the times the model predicted class X, how often was it actually class X? High precision = few false alarms. |
| **Recall** | Of all the actual class X images, how many did the model correctly identify? High recall = few missed cases. |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization. A technique to enhance local contrast in images. |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping. Highlights which pixels of the input image most influenced the model's prediction. Used for explainability. |
| **Checkpoint (.pt file)** | A saved copy of a trained model's weights. Loading a checkpoint restores the model to its trained state. |
| **Held-out test set** | Data that is locked away and never used during training or any decision-making. Used only at the very end to measure real-world performance. |
| **Macro-F1** | Average F1 score across all classes, treating each class equally regardless of how many images it has. |
| **Early stopping** | Automatically stopping training when the model stops improving on validation data. Prevents overfitting. |
| **Overfitting** | When a model memorises the training data and performs worse on new data. Good validation scores that match test scores indicate no overfitting. |
| **Bounding box** | A rectangle drawn around an object in an image. Defined by coordinates `[x1, y1, x2, y2]`. |
| **Router** | In this project: the first model that decides if an input image is a leaf or a fruit photo before sending it to the appropriate classifier. |
| **IoU** | Intersection over Union. Measures how much two bounding boxes overlap. IoU = (overlap area) / (union area). |
| **YOLO** | "You Only Look Once" — a family of fast object detection models. |
| **Gradio** | A Python library for building simple web interfaces for machine learning models. Used for the demo app in this project. |

---

*End of report. For questions, contact the original developer or refer to the commit history on the `4-class-classifier` branch.*
