# Project Report — Version 1
## Tomato Disease Diagnosis System (Full Pipeline)

**Branch:** `main`  
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
10. [The Gradio web interface](#10-the-gradio-web-interface)
11. [Test results — all numbers explained](#11-test-results--all-numbers-explained)
12. [How to set up and run from scratch](#12-how-to-set-up-and-run-from-scratch)
13. [Known limitations](#13-known-limitations)
14. [Glossary](#14-glossary)

---

## 1. What this project does

This system takes a photo of a tomato plant and automatically diagnoses what condition it is in.

**It classifies tomato images into 18 possible conditions across two domains:**

### Leaf diseases (10 classes)

| Class | Description |
|---|---|
| Bacterial spot | Small water-soaked dark spots on leaves |
| **Early blight** | Brown target-ring lesions (Alternaria solani) |
| **Healthy** | No disease — healthy green leaf |
| Late blight | Dark water-soaked patches (Phytophthora infestans) |
| Leaf Mold | Yellow patches, grey mold under the leaf |
| Septoria leaf spot | Small circular spots with dark margins |
| Spider mites (Two-spotted) | Fine webbing, tiny yellow speckling |
| Target Spot | Brown circular lesions |
| Tomato Yellow Leaf Curl Virus | Curled, yellowing leaves |
| Tomato mosaic virus | Mosaic yellowing/mottling pattern |

### Fruit disorders (8 classes)

| Class | Description |
|---|---|
| Anthracnose | Dark sunken circular spots on ripe fruit |
| **Blossom_End_Rot** | Dark sunken patch at the blossom end (calcium deficiency) |
| Catfaced | Deformed fruit with scarring and cavities |
| Fruit_Cracking | Radial or concentric cracks in the skin |
| **Healthy_Tomato** | Normal healthy fruit |
| Late_Blight | Dark water-soaked lesions on the fruit surface |
| Mold | Visible mould growth on the fruit |
| Spotted_Wilt_Virus | Circular spots, bronzing, ring patterns |

**Additional outputs for every classification:**
- Confidence score (0–100%)
- Disease severity percentage (how much of the leaf or fruit is affected)
- Grad-CAM heatmap (leaf only) — shows which region of the image triggered the prediction
- Bounding boxes (fruit only) — shows where each defect is located
- Unified JSON output for programmatic use

> **Note:** A four-class simplified variant (Early blight, Blossom_End_Rot, Healthy leaf, Healthy plant) is available on the `4-class-classifier` branch.

---

## 2. Why it was built this way

### The core design challenge

Tomato disease manifests in two completely different visual forms:

- **Leaf diseases** appear as colour patterns, texture changes, or spots on a flat leaf surface. This is a classic **image classification** problem — one label per image.
- **Fruit disorders** involve specific regions on a 3D fruit surface. The fruit dataset arrived with **bounding box annotations** (annotators drew rectangles around each defect). Collapsing this to single-label classification would discard the location and size information. Object detection is the right tool here.

A single model architecture cannot handle both optimally. The solution is a **three-model pipeline**:

```
Router → decides: "leaf or fruit?"
  ├── Leaf  → EfficientNet-B0 classifier (10 classes)
  └── Fruit → YOLOv8n detector (8 classes)
```

### Why transfer learning instead of training from scratch?

- The GPU available (RTX 3050 Ti, 4GB VRAM) is mid-range
- Training from scratch requires orders of magnitude more data and compute
- EfficientNet-B0 and MobileNetV3 pre-trained on ImageNet already understand general visual features (edges, textures, shapes)
- YOLOv8n pre-trained on COCO already understands object detection
- Fine-tuning adapts these features to tomato disease with far fewer images and epochs

### Why two-phase training for the leaf classifier?

Phase 1 trains only the new classification head (last layer). Phase 2 unfreezes the last feature blocks and fine-tunes them at a low learning rate. This prevents the backbone from being corrupted by random gradients early on, while still allowing it to adapt its features to tomato leaf patterns.

---

## 3. Architecture and system design

---

### 3.1 High-level system overview

```
╔══════════════════════════════════════════════════════════════════════╗
║              TOMATO DISEASE DIAGNOSIS SYSTEM (main branch)           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ┌──────────────┐      ┌─────────────────────────────────────────┐ ║
║   │   Web UI     │      │            INFERENCE PIPELINE            │ ║
║   │  (Gradio)    │─────▶│                                         │ ║
║   └──────────────┘      │  ┌──────────┐   leaf  ┌──────────────┐  │ ║
║                         │  │  Router  │────────▶│  Leaf Model  │  │ ║
║   ┌──────────────┐      │  │MobileNet │         │ EfficientNet │  │ ║
║   │  CLI / API   │─────▶│  │  V3-Sm   │         │     B0       │  │ ║
║   │  Python      │      │  │          │  fruit  └──────┬───────┘  │ ║
║   └──────────────┘      │  │leaf vs   │         ┌──────▼───────┐  │ ║
║                         │  │fruit     │────────▶│ Fruit Model  │  │ ║
║                         │  └──────────┘         │  YOLOv8n     │  │ ║
║                         │                       └──────┬───────┘  │ ║
║                         └──────────────────────────────┼───────────┘ ║
║                                                        │             ║
║   ┌────────────────────────────────────────────────────▼───────────┐ ║
║   │                  EXPLAINABILITY + SEVERITY                      │ ║
║   │  Leaf  → Grad-CAM heatmap  +  HSV/Lab severity %               │ ║
║   │  Fruit → YOLO bounding boxes  +  box-geometry severity %       │ ║
║   └────────────────────────────────────────────────────┬───────────┘ ║
║                                                        │             ║
║   ┌────────────────────────────────────────────────────▼───────────┐ ║
║   │                      OUTPUT                                     │ ║
║   │  domain · class · confidence · severity_pct ·                   │ ║
║   │  gradcam_overlay · detections · model_version                  │ ║
║   │  + saved result images + JSON log                              │ ║
║   └─────────────────────────────────────────────────────────────── ┘ ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

### 3.2 Full inference data flow

```
                          INPUT IMAGE
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Image Preprocessing   │
                 │  • Open (PIL)           │
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
                 │  Output: [p_leaf,       │
                 │           p_fruit]      │
                 └──────┬─────────┬───────┘
                        │         │
                  p_leaf wins   p_fruit wins
                        │         │
        ┌───────────────▼─┐    ┌──▼────────────────────┐
        │   LEAF BRANCH    │    │     FRUIT BRANCH       │
        │                  │    │                        │
        │  EfficientNet-B0 │    │  YOLOv8n               │
        │  10-class softmax│    │  640×640 input         │
        │                  │    │  Output: boxes         │
        │  → class name    │    │  [cls, conf, x1,y1,    │
        │  → confidence    │    │   x2,y2]               │
        │                  │    │                        │
        │  Grad-CAM:       │    │  Primary class =       │
        │  • hooks on      │    │  highest-conf box      │
        │    features[-1]  │    │                        │
        │  • backward pass │    │  Severity = disease    │
        │  • heatmap saved │    │  box area / fruit area │
        │    to disk       │    │                        │
        │                  │    │                        │
        │  Severity:       │    │                        │
        │  HSV canopy mask │    │                        │
        │  + Lab deviation │    │                        │
        └──────┬───────────┘    └──────────┬─────────────┘
               │                           │
               └──────────────┬────────────┘
                              │
                              ▼
             ┌─────────────────────────────────┐
             │         UNIFIED JSON OUTPUT      │
             │  {                               │
             │    "domain":          "leaf",    │
             │    "class":           "...",     │
             │    "confidence":      0.94,      │
             │    "severity_pct":    4.42,      │
             │    "gradcam_overlay": "path",    │
             │    "detections":      null,      │
             │    "model_version":   "..."      │
             │  }                               │
             └────────────────┬────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
     ┌──────────────────┐        ┌──────────────────────┐
     │   Gradio UI      │        │   File System         │
     │   (browser)      │        │                       │
     │  3 columns:      │        │  *_result.jpg         │
     │  • upload        │        │  *_explain.jpg        │
     │  • Grad-CAM      │        │  inference_log.json   │
     │    / YOLO boxes  │        │  gradcam/*.jpg        │
     │  • score panel   │        └──────────────────────┘
     └──────────────────┘
```

---

### 3.3 Training pipeline design

```
  ┌──────────────────────┐    ┌──────────────────────────────┐
  │    PlantVillage       │    │   Roboflow Fruit Dataset      │
  │  ~16k tomato images  │    │  1319 images + YOLO labels    │
  │  10 folders/classes  │    │  (~440 unique source images,  │
  └──────────┬───────────┘    │   Roboflow made 3 versions    │
             │                │   of each)                    │
             ▼                └──────────────┬────────────────┘
  ┌──────────────────────┐                   │
  │ split_leaf_           │                   ▼
  │ manifest.py          │    ┌──────────────────────────────┐
  │                      │    │  split_fruit_groups.py        │
  │ Stratified 70/15/15  │    │                              │
  │ seed=42, by class    │    │  Groups by Roboflow source   │
  │                      │    │  stem before splitting →     │
  │ → train.csv (11208)  │    │  no augmentation leakage    │
  │   val.csv   (2402)   │    │                              │
  │   test.csv  (2401)   │    │ → images/train|val|test/     │
  │   class_map.json     │    │   labels/train|val|test/     │
  └──────────┬───────────┘    │   data.yaml                  │
             │                └──────────────┬────────────────┘
             ▼                               │
  ┌──────────────────────┐                   │
  │ build_router_         │                   │
  │ manifest.py          │                   │
  │                      │                   │
  │ Merges leaf + fruit   │                   │
  │ with domain label    │                   │
  │ → router/            │                   │
  │   train|val|test.csv │                   │
  └──────────┬───────────┘                   │
             │                               │
  ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ TRAINING ─
             ▼                               ▼
  ┌──────────────────────────┐  ┌────────────────────────────┐
  │  train.py                │  │  train_fruit_yolo.py        │
  │  (router + leaf)         │  │  (Ultralytics YOLOv8)       │
  │                          │  │                            │
  │  ── ROUTER ──            │  │  Pretrained: yolov8n.pt    │
  │  MobileNetV3-Small       │  │  (COCO weights)            │
  │  Phase 1: 3 epochs       │  │                            │
  │  Phase 2: 5 epochs       │  │  Fine-tune:                │
  │  Equal domain sampling   │  │  100 epochs max            │
  │                          │  │  patience = 20             │
  │  ── LEAF ──              │  │  image size = 640          │
  │  EfficientNet-B0         │  │  batch = 8                 │
  │  Phase 1: head only      │  │                            │
  │    10 epochs, lr=0.001   │  │  → runs/fruit/train/       │
  │  Phase 2: unfreeze 3blk  │  │    weights/best.pt         │
  │    15 epochs, lr=0.00005 │  └──────────────┬────────────┘
  │  Early stop patience=4   │                 │
  └──────────────┬───────────┘                 │
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
          ┌─────────────────────────────────────────┐
          │             checkpoints/                  │
          │                                          │
          │  router_v1_mobilenet_v3_small.pt         │
          │  leaf_v1_efficientnet_b0.pt              │
          │  fruit_v1_yolov8n.pt                     │
          └─────────────────────────────────────────┘
```

---

### 3.4 Model architecture details

#### Router — MobileNetV3-Small

```
Input: 224×224×3 RGB image
    │
    ▼  CLAHE → resize → normalise → tensor
    │
    ▼
┌─────────────────────────────────────────────┐
│        MobileNetV3-Small backbone            │
│  Pre-trained on ImageNet, frozen in Phase 1  │
│  Inverted residual blocks with               │
│  Squeeze-and-Excitation attention            │
│  Output features: 576 channels              │
└────────────────────┬────────────────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │  AdaptiveAvgPool2d      │
         │  Hardswish activation   │
         │  Linear(576  → 1024)    │
         │  Hardswish activation   │
         │  Dropout(p=0.2)         │
         │  Linear(1024 → 2)       │  ← replaced head
         └────────────┬────────────┘
                      ▼
               Softmax → [p_leaf, p_fruit]
               Argmax  → 0=leaf or 1=fruit
```

#### Leaf Classifier — EfficientNet-B0

```
Input: 224×224×3 RGB image
    │
    ▼  CLAHE → resize → augment(train) → normalise
    │
    ▼
┌─────────────────────────────────────────────┐
│           EfficientNet-B0 backbone           │
│  Pre-trained on ImageNet                     │
│                                             │
│  features[0]  Conv stem                    │
│  features[1]  MBConv ×1  (16ch)            │
│  features[2]  MBConv ×2  (24ch)            │
│  features[3]  MBConv ×2  (40ch)            │
│  features[4]  MBConv ×3  (80ch)            │
│  features[5]  MBConv ×3  (112ch)           │
│  features[6]  MBConv ×4  (192ch) ◀ Grad-CAM target
│  features[7]  MBConv ×1  (320ch)           │
│  features[8]  Conv head  (1280ch)           │
│                                             │
│  Phase 1 training: only classifier trained │
│  Phase 2 training: features[6][7][8]+head  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
         ┌─────────────────────────┐
         │  AdaptiveAvgPool2d      │
         │  Dropout(p=0.2)         │
         │  Linear(1280 → 10)      │  ← replaced head
         └────────────┬────────────┘
                      ▼
           Softmax → 10 probabilities
           Argmax → predicted class (0–9)
```

#### Fruit Detector — YOLOv8n

```
Input: any image (auto-resized to 640×640)
    │
    ▼
┌─────────────────────────────────────────────┐
│              YOLOv8n Backbone                │
│  Pre-trained on COCO                         │
│  CSP-Darknet + depthwise separable convs    │
│  C2f blocks (cross-stage partial network)   │
│  SPPF (Spatial Pyramid Pooling Fast)        │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│                  Neck                        │
│  FPN (Feature Pyramid Network):             │
│    fuses features from 3 scales             │
│  PAN (Path Aggregation Network):            │
│    strengthens localisation signals         │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│          Detection Head (3 scales)           │
│  Each point outputs:                        │
│  [x, y, w, h, objectness, cls_probs×8]     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
       NMS: conf_threshold=0.25, iou=0.70
                     │
                     ▼
       Detections: [class_id, conf, x1, y1, x2, y2]
       Primary class = highest confidence box
```

---

### 3.5 Grad-CAM explainability architecture

```
  Input tensor [1, 3, 224, 224]
      │
      ▼  Register hooks at features[-1]  (last EfficientNet block)
      │
      ├──────── Forward pass ──────────────────────────────────┐
      │                                                        │
      ▼                                              Activations A
  Logits → score for predicted class                [1, C, H, W]
      │
      ▼  Backward pass (gradients flow back)
      │
      └──────── Backward hook captures ───────────────────────┐
                                                              │
                                                    Gradients G
                                                    [1, C, H, W]
      │
      ▼  weights w = mean(G over spatial dims)  →  [C, 1, 1]
      │
      ▼  CAM = ReLU( sum(w × A, over channels) )  →  [H, W]
      │
      ▼  Resize to 224×224,  Normalize [0, 1]
      │
      ▼  cv2.COLORMAP_JET  →  blue (ignored) to red (important)
      │
      ▼  Blend with original image (alpha=0.45)
      │
      ▼  Save to results/gradcam_overlays/<stem>_gradcam.jpg
```

---

### 3.6 Severity scoring design

```
  LEAF SEVERITY (colour science approach)
  ────────────────────────────────────────
  Input: RGB image
    │
    ▼  Convert to HSV
  Canopy mask = pixels where Hue ∈ [25°,95°], Sat>40, Val>40
    │
    ▼  Convert to Lab colour space
  For each canopy pixel:
    deviation = |a − median_a| + |b − median_b|
    (a,b channels = colour shift from healthy green)
    │
    ▼  Threshold: deviation > 18.0
  Lesion pixels = canopy pixels with high colour shift
    │
    ▼
  severity_pct = (lesion pixels / canopy pixels) × 100


  FRUIT SEVERITY (bounding box geometry)
  ────────────────────────────────────────
  Input: YOLO detections + image dimensions
    │
    ▼
  disease_area = Σ (w × h) for each non-healthy box
    │
  estimated_fruit_area:
    • if Healthy_Tomato box present → use that box area
    • else if multiple boxes → union of all boxes
    • else single disease box → full image area (fallback)
    │
    ▼
  severity_pct = min(100, disease_area / fruit_area × 100)
```

---

### 3.7 Data storage design

```
  checkpoints/          ← Trained weights (binary, ~43 MB, not in git)
  │
  ├── router_v1_mobilenet_v3_small.pt
  │     {model_state_dict, config, val_metrics, phase, epoch}
  │
  ├── leaf_v1_efficientnet_b0.pt
  │     {model_state_dict, config, class_names[10], val_metrics,
  │      phase, epoch, metrics_history, best_macro_f1,
  │      completed_phases, best_phase, best_epoch}
  │
  └── fruit_v1_yolov8n.pt
        Ultralytics YOLO standard format
        Loaded via: YOLO(path).predict(image, conf=0.25, iou=0.7)

  data/splits/          ← Data manifests (CSV/JSON, committed to git)
  │
  ├── leaf/
  │     train.csv       11,208 rows: image_path, label
  │     val.csv          2,402 rows
  │     test.csv         2,401 rows
  │     class_map.json  {class_names[10], class_to_idx{...}}
  │
  ├── fruit/
  │     data.yaml       YOLO format (path, train, val, test, nc, names)
  │     labels/train/*.txt   YOLO box annotations (committed)
  │     labels/val/*.txt
  │     labels/test/*.txt
  │
  └── router/
        train.csv       12,118 rows: image_path, domain, domain_idx
        val.csv          2,601 rows
        test.csv         2,611 rows
        domain_map.json {class_names, domain_to_idx}

  results/              ← Reports, Grad-CAM images, inference results
  │
  ├── leaf_classifier_test_report.json   ← per-class F1, confusion matrix
  ├── leaf_classifier_test_report.md
  ├── fruit_detector_test_report.json    ← per-class mAP
  ├── fruit_detector_test_report.md
  ├── project_summary.json
  ├── gradcam_overlays/                 ← Grad-CAM images (created at runtime)
  └── inference_results/               ← Saved by app.py at runtime
        inference_log.json             ← Appended every run
        *_result.jpg                   ← Annotated score panel
        *_explain.jpg                  ← Grad-CAM or detection overlay
        gradcam/                       ← Grad-CAM images from app
```

---

### 3.8 Component interaction map

```
  app.py
    └── imports infer_pipeline.py     (TomatoInferencePipeline)

  infer_pipeline.py
    ├── imports evaluate.py           (load_leaf_checkpoint,
    │                                  load_router_checkpoint, repo_root)
    ├── imports gradcam.py            (compute_gradcam, save_gradcam_overlay)
    ├── imports models/leaf_classifier.py
    ├── imports models/router.py
    ├── imports severity.py           (FruitDetection, leaf_severity_pct,
    │                                  fruit_severity_pct)
    ├── imports transforms.py         (build_leaf_transforms)
    └── imports train_fruit_yolo.py   (resolve_data_yaml,
                                       write_resolved_data_yaml)

  evaluate.py
    ├── imports models/leaf_classifier.py  (build_leaf_classifier)
    └── imports models/router.py           (build_router)

  train.py
    ├── imports datasets.py           (build_leaf_dataloader,
    │                                  build_router_dataloader)
    ├── imports models/leaf_classifier.py
    ├── imports models/router.py
    └── imports transforms.py

  datasets.py
    └── imports transforms.py         (build_leaf_transforms)
```

---

### 3.9 Technology stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ | All code |
| **Deep learning** | PyTorch | ≥ 2.6.0 | Model training and inference |
| **Vision models** | torchvision | ≥ 0.21.0 | EfficientNet-B0, MobileNetV3 |
| **Object detection** | Ultralytics | ≥ 8.3.107 | YOLOv8n training + inference |
| **Image processing** | OpenCV (cv2) | ≥ 4.11.0 | CLAHE, HSV masking, Grad-CAM |
| **Image I/O** | Pillow (PIL) | ≥ 10.0.0 | Loading, converting, saving images |
| **Array math** | NumPy | ≥ 2.1.1 | Heatmap computation, severity |
| **Config** | PyYAML | ≥ 6.0 | Training configs |
| **Web UI** | Gradio | ≥ 4.0.0 | Browser-based demo interface |
| **Hardware** | NVIDIA RTX 3050 Ti | 4GB VRAM | Training (GPU) |

---

## 4. Repository structure — every file explained

```
IIHR-object-detection-and-classification/
│
├── src/
│   ├── app.py                    ← Gradio web interface (this file)
│   ├── infer_pipeline.py         ← Main inference: router → leaf/fruit → JSON
│   ├── train.py                  ← Training: router + leaf classifier
│   ├── train_fruit_yolo.py       ← Training: fruit YOLOv8n (Ultralytics)
│   ├── evaluate.py               ← Evaluation utilities + checkpoint loaders
│   ├── datasets.py               ← PyTorch Dataset classes (leaf + router)
│   ├── transforms.py             ← Image preprocessing pipeline (CLAHE etc.)
│   ├── gradcam.py                ← Grad-CAM explainability
│   ├── severity.py               ← Leaf HSV/Lab + fruit box-geometry severity
│   ├── split_leaf_manifest.py    ← Generates leaf train/val/test CSVs
│   ├── split_fruit_groups.py     ← Group-aware fruit YOLO split
│   ├── build_router_manifest.py  ← Generates router train/val/test CSVs
│   └── models/
│       ├── leaf_classifier.py    ← EfficientNet-B0 definition + phase helper
│       └── router.py             ← MobileNetV3-Small definition
│
├── configs/
│   ├── leaf.yaml                 ← EfficientNet-B0 training params
│   ├── router.yaml               ← MobileNetV3-Small training params
│   └── fruit_yolo.yaml           ← YOLOv8n training params
│
├── checkpoints/
│   ├── README.md                 ← How to obtain the .pt weight files
│   ├── router_v1_mobilenet_v3_small.pt  (binary — train locally)
│   ├── leaf_v1_efficientnet_b0.pt       (binary — train locally)
│   └── fruit_v1_yolov8n.pt              (binary — train locally)
│
├── data/
│   ├── README.md                 ← Data download + setup instructions
│   └── splits/
│       ├── leaf/                 ← train.csv, val.csv, test.csv, class_map.json
│       ├── fruit/                ← data.yaml, labels/, (images/ not in git)
│       └── router/               ← train.csv, val.csv, test.csv, domain_map.json
│
├── results/
│   ├── leaf_classifier_test_report.json
│   ├── leaf_classifier_test_report.md
│   ├── fruit_detector_test_report.json
│   ├── fruit_detector_test_report.md
│   ├── project_summary.json
│   └── inference_results/        ← Created at runtime by app.py
│       ├── inference_log.json
│       └── *.jpg
│
├── scripts/
│   ├── eval_leaf_test_report.py        ← Reproduces leaf test metrics
│   ├── eval_fruit_test_report.py       ← Reproduces fruit test metrics
│   ├── verify_leaf_checkpoint.py       ← Integrity check on leaf .pt
│   ├── analyze_early_blight_confusion.py
│   ├── analyze_fruit_confusion.py
│   └── analyze_swv_box_areas.py
│
├── report_version1.md            ← THIS FILE
├── README.md                     ← Quick start guide
├── ARCHITECTURE.md               ← Design decisions
├── requirements.txt              ← Python dependencies
└── .gitignore
```

---

## 5. The datasets

### Dataset 1 — Leaf (PlantVillage)

- **Source:** Kaggle `emmarex/plantdisease`, filtered to tomato-only
- **Type:** Image classification (one folder per class)
- **Total:** ~16,000 images across 10 classes
- **Local path (not in git):** `data/leaf_raw/PlantVillage/`
- **Split:** Stratified 70 / 15 / 15, seed 42

| Index | Class | Test images |
|---|---|---:|
| 0 | Bacterial spot | 319 |
| 1 | Early blight | 150 |
| 2 | Healthy | 238 |
| 3 | Late blight | 287 |
| 4 | Leaf Mold | 143 |
| 5 | Septoria leaf spot | 265 |
| 6 | Spider mites (Two-spotted) | 252 |
| 7 | Target Spot | 210 |
| 8 | Tomato Yellow Leaf Curl Virus | 481 |
| 9 | Tomato mosaic virus | 56 |
| **Total** | | **2,401** |

### Dataset 2 — Fruit (Roboflow)

- **Source:** Kaggle `profnourasemary/tomato-dataset` (Roboflow YOLOv8 export)
- **Type:** Object detection (YOLO bounding box format)
- **Total:** 1,319 images (but ~440 unique sources — Roboflow created 3 augmented copies each)
- **Local path (images not in git):** `data/fruit_raw/`
- **Split:** Group-aware 70 / 15 / 15

| ID | Class | Description |
|---|---|---|
| 0 | Anthracnose | Dark sunken spots |
| 1 | Blossom_End_Rot | Calcium-deficiency patch |
| 2 | Catfaced | Deformed fruit |
| 3 | Fruit_Cracking | Skin cracks |
| 4 | Healthy_Tomato | No disease |
| 5 | Late_Blight | Water-soaked fruit lesions |
| 6 | Mold | Visible mould growth |
| 7 | Spotted_Wilt_Virus | Ring spots, bronzing |

**Important — augmentation leakage problem:** Roboflow baked 3 augmented versions of each source image. A naive random split puts augmented copies of the same image in both training and test — inflating test scores artificially. `split_fruit_groups.py` solves this by grouping images by source stem (the identifier before `.rf.` in the filename) and splitting at the group level.

---

## 6. Data preprocessing

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Applied to every image before it enters any model.

- Works in LAB colour space — only the L (lightness) channel is modified
- Enhances local contrast in small regions (tile grid 8×8)
- Clip limit of 2.0 prevents over-amplifying noise
- Makes disease patterns more distinct under variable lighting conditions
- Implemented in `transforms.py → CLAHETransform`

### Leaf training transforms

```
Training:   CLAHE → Resize(224×224) → RandomHorizontalFlip(p=0.5)
            → RandomRotation(±15°) → ColorJitter(brightness/contrast
            /saturation/hue) → ToTensor → Normalize(ImageNet)

Inference:  CLAHE → Resize(224×224) → ToTensor → Normalize(ImageNet)
```

**ImageNet normalisation values:** mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

### Fruit group-aware splitting

`split_fruit_groups.py` extracts the source stem from each filename (the part before `.rf.` which Roboflow appends). All images sharing the same stem are assigned to the same split. The split is done at group level, 70/15/15. This guarantees no augmentation leakage.

---

## 7. The three models

### Model 1 — Router (MobileNetV3-Small)

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Small |
| Pretrained on | ImageNet |
| Input size | 224×224 |
| Output classes | 2 (leaf=0, fruit=1) |
| Training phases | 2 (head only → unfreezing) |
| Checkpoint | `router_v1_mobilenet_v3_small.pt` |
| Test accuracy | **100%** |

The router is a lightweight binary classifier. MobileNetV3-Small was chosen because the task (leaf vs fruit is visually obvious) doesn't require a powerful model. Being small means fast routing with minimal GPU overhead.

### Model 2 — Leaf Classifier (EfficientNet-B0)

| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Pretrained on | ImageNet |
| Input size | 224×224 |
| Output classes | 10 leaf disease classes |
| Phase 1 | Train head only. 10 epochs max, lr=0.001, frozen backbone |
| Phase 2 | Unfreeze last 3 blocks. 15 epochs max, lr=0.00005 |
| Early stopping | Patience=4 on val macro-F1 |
| Class imbalance | Inverse-frequency weights in CrossEntropyLoss |
| Checkpoint | `leaf_v1_efficientnet_b0.pt` |
| Best val macro-F1 | **0.9930** |
| Test macro-F1 | **0.9942** |

### Model 3 — Fruit Detector (YOLOv8n)

| Property | Value |
|---|---|
| Architecture | YOLOv8n (nano) |
| Pretrained on | COCO |
| Input size | 640×640 |
| Output classes | 8 fruit defect classes |
| Max epochs | 100 |
| Early stopping | Patience=20 |
| Best epoch | 44 (stopped at 64) |
| Checkpoint | `fruit_v1_yolov8n.pt` |
| Test mAP@0.5 | **0.807** |

---

## 8. How training works

### Generating data splits

```powershell
# Requires PlantVillage data at data/leaf_raw/PlantVillage/
python src/split_leaf_manifest.py --write

# Requires fruit data at data/fruit_raw/
python src/split_fruit_groups.py

# Requires leaf splits to exist
python src/build_router_manifest.py --write
```

### Training the models (in order)

```powershell
# 1. Router (trains in minutes)
python src/train.py --config configs/router.yaml

# 2. Leaf classifier (Phase 1 ~10 epochs + Phase 2 ~12 epochs)
python src/train.py --config configs/leaf.yaml

# 3. Fruit YOLO (best epoch 44, stops at 64)
python src/train_fruit_yolo.py --config configs/fruit_yolo.yaml
```

### What `train.py` does internally

1. Loads CSV manifests via `datasets.py → LeafDataset` or `RouterDataset`
2. Applies `transforms.py` augmentation pipeline
3. Builds model from `src/models/`
4. **Phase 1:** Freezes backbone, trains only the new head. Monitors val macro-F1.
5. **Phase 2:** Unfreezes last N feature blocks, resumes with lower LR.
6. **Early stopping:** If val macro-F1 doesn't improve by >0.003 for 4 epochs, stops.
7. Saves best checkpoint to `checkpoints/` including full metrics history.

### Checkpoint contents (leaf)

```python
{
    "model_state_dict":  ...,     # Trained weights tensor dict
    "config":            ...,     # Full YAML config as dict
    "class_names":       [...],   # ["Bacterial spot", "Early blight", ...]
    "val_metrics":       {...},   # Best val precision/recall/F1/confusion
    "phase":             2,       # Which phase achieved best score
    "epoch":             12,      # Which epoch within that phase
    "best_phase":        2,
    "best_epoch":        12,
    "metrics_history":   [...],   # Full training curve
    "best_macro_f1":     0.9930,
    "completed_phases":  [1, 2],
}
```

---

## 9. How inference works

### Running inference

```powershell
# CLI — pretty printed
python src/infer_pipeline.py path\to\image.jpg

# CLI — JSON output only
python src/infer_pipeline.py --json path\to\image.jpg

# Multiple images
python src/infer_pipeline.py --json img1.jpg img2.jpg
```

### What `TomatoInferencePipeline.infer()` does

1. Opens the image with PIL, converts to RGB
2. Runs the **router** → determines `"leaf"` or `"fruit"`
3. **Leaf branch:**
   - Runs `leaf_model` forward pass → 10-class softmax
   - Picks argmax → predicted class + confidence
   - Computes Grad-CAM → saves overlay to `results/gradcam_overlays/`
   - Computes HSV/Lab severity percentage
4. **Fruit branch:**
   - Runs `YOLO.predict()` → list of detections
   - Primary class = highest-confidence detection
   - If no detections → defaults to `Healthy_Tomato`
   - Computes box-geometry severity percentage
5. Returns a unified dict with all fields

### Unified JSON output schema

```json
{
  "domain":          "leaf | fruit",
  "class":           "Early blight",
  "confidence":      0.9446,
  "severity_pct":    4.42,
  "gradcam_overlay": "results/gradcam_overlays/image_gradcam.jpg",
  "detections":      null,
  "model_version":   "leaf_v1_efficientnet_b0"
}
```

For fruit, `detections` is an array:
```json
"detections": [
  {"class": "Blossom_End_Rot", "confidence": 0.693, "bbox": [369.0, 94.5, 525.3, 307.2]},
  {"class": "Catfaced",        "confidence": 0.255, "bbox": [0.1, 0.0, 160.1, 289.7]}
]
```

---

## 10. The Gradio web interface

**File:** `src/app.py`  
**Run:** `python src/app.py`  
**Opens:** `http://127.0.0.1:7860`

### Layout (3 columns)

```
┌───────────────┬──────────────────────┬──────────────────────┐
│  Upload       │  Grad-CAM / YOLO     │  Score panel         │
│               │  detection overlay   │                      │
│  [drag&drop]  │  (explainability)    │  Predicted class     │
│               │                      │  Confidence bar      │
│  [Diagnose]   │  Leaf: heatmap       │  Severity %          │
│               │  Fruit: bounding     │  All detections      │
│  Leaf classes │  boxes on original   │                      │
│  [accordion]  │                      │  Domain + timestamp  │
│               │                      │                      │
│  Fruit classes│                      │                      │
│  [accordion]  │                      │                      │
└───────────────┴──────────────────────┴──────────────────────┘
│  Markdown result summary (class, confidence, severity)       │
│  Saved file paths                                            │
└──────────────────────────────────────────────────────────────┘
```

### Auto-saved files per run

| File | Contents |
|---|---|
| `results/inference_results/YYYYMMDD_HHMMSS_<class>_result.jpg` | Annotated score panel |
| `results/inference_results/YYYYMMDD_HHMMSS_<class>_explain.jpg` | Grad-CAM or detection overlay |
| `results/inference_results/inference_log.json` | Running log (appended) |
| `results/inference_results/gradcam/*.jpg` | Grad-CAM images used in the app |

---

## 11. Test results — all numbers explained

All numbers come from the **held-out test set** — data never used in training or checkpoint selection.

### Router

| Split | Accuracy |
|---|---:|
| Validation | **100.00%** |
| Test | **100.00%** |

Zero routing errors on both splits.

### Leaf Classifier — all 10 classes

| Class | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Bacterial spot | 0.9969 | 0.9969 | 0.9969 | 319 |
| **Early blight** | **1.0000** | **0.9800** | **0.9899** | 150 |
| **Healthy** | **1.0000** | **1.0000** | **1.0000** | 238 |
| Late blight | 0.9930 | 0.9930 | 0.9930 | 287 |
| Leaf Mold | 0.9930 | 0.9930 | 0.9930 | 143 |
| Septoria leaf spot | 0.9925 | 1.0000 | 0.9962 | 265 |
| Spider mites (Two-spotted) | 0.9804 | 0.9921 | 0.9862 | 252 |
| Target Spot | 0.9905 | 0.9905 | 0.9905 | 210 |
| TYLCV | 0.9979 | 0.9938 | 0.9958 | 481 |
| Tomato mosaic virus | 1.0000 | 1.0000 | 1.0000 | 56 |
| **Overall macro-F1** | | | **0.9942** | **2,401** |

**Val vs test comparison:** Test macro-F1 (0.9942) is **above** val best (0.9930) — no overfitting. No class dropped more than 0.004 F1 from val to test.

**Early blight precision history:**

| Stage | Precision | Late blight → EB false positives |
|---|---:|---:|
| Phase 1 baseline | 0.7444 | 19 |
| Phase 2 val (best) | 0.9804 | 1 |
| **Test (held-out)** | **1.0000** | **0** |

### Fruit Detector — all 8 classes

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Anthracnose | 0.779 | 0.923 | 0.851 | 0.475 |
| **Blossom_End_Rot** | **0.862** | **0.923** | **0.918** | 0.566 |
| Catfaced | 0.785 | 0.850 | 0.830 | 0.394 |
| Fruit_Cracking | 0.698 | 0.576 | 0.689 | 0.450 |
| **Healthy_Tomato** | **0.824** | **0.618** | **0.760** | 0.502 |
| Late_Blight | 0.875 | 0.762 | 0.865 | 0.538 |
| Mold | 0.778 | 0.561 | 0.688 | 0.333 |
| Spotted_Wilt_Virus | 0.882 | 0.708 | 0.853 | 0.514 |
| **Overall mAP@0.5** | | | **0.807** | 0.471 |

**Known issues:**
- **Catfaced** (val): precision=0.568, recall=1.0 — over-detects (confuses Fruit_Cracking as Catfaced)
- **Spotted_Wilt_Virus**: low recall (0.37 val) — insufficient labeled data + confusion with Late_Blight

### Summary

| Component | Metric | Score |
|---|---|---:|
| Router | Accuracy | **1.0000** |
| Leaf classifier | Macro-F1 | **0.9942** |
| Fruit detector | mAP@0.5 | **0.8067** |

---

## 12. How to set up and run from scratch

### Step 1 — Clone

```powershell
git clone https://github.com/Gowthamjignas06/IIHR-object-detection-and-classification.git
cd IIHR-object-detection-and-classification
```

### Step 2 — Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

With NVIDIA GPU:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Step 3 — Get model weights

The `.pt` files are not in git (~43 MB total). Either:

**Option A — Train locally:**
```powershell
# Download PlantVillage → data/leaf_raw/PlantVillage/
# Download Roboflow fruit data → data/fruit_raw/

python src/split_leaf_manifest.py --write
python src/split_fruit_groups.py
python src/build_router_manifest.py --write

python src/train.py --config configs/router.yaml
python src/train.py --config configs/leaf.yaml
python src/train_fruit_yolo.py --config configs/fruit_yolo.yaml
```

**Option B — Get from developer:** Place the three `.pt` files in `checkpoints/`.

### Step 4 — Launch the web interface

```powershell
python src/app.py
```

Opens `http://127.0.0.1:7860`.

### Step 5 — CLI inference

```powershell
python src/infer_pipeline.py --json path\to\image.jpg
```

### Step 6 — Evaluate models

```powershell
python scripts/eval_leaf_test_report.py
python scripts/eval_fruit_test_report.py
python scripts/verify_leaf_checkpoint.py
```

---

## 13. Known limitations

| Limitation | Detail | Severity |
|---|---|---|
| Catfaced over-detection | High recall (1.0 val) but low precision (0.57 val) — confuses Fruit_Cracking and background spurious boxes with Catfaced | Medium |
| Spotted_Wilt_Virus low recall | Val recall 0.37 — insufficient labeled examples, visual confusion with Late_Blight | Medium |
| Fruit severity approximation | Box-geometry heuristic sensitive to photo cropping distance | Low |
| Healthy_Tomato recall | 0.62 on test — YOLO sometimes doesn't draw a box around a healthy tomato when fruit is small or off-center | Medium |
| Out-of-scope images | The system always outputs a class — it cannot say "I don't know this plant" | Low for intended use |
| No hardware integration | Phase 3 (sensors, actuators, fuzzy logic dosing) is not implemented | Deferred by design |
| Image quality sensitivity | Best results when leaf/fruit fills most of the frame. HEIC not supported. | Low |

### Out of scope (Phase 3 — deferred)

GPIO/ADC sensors, fuzzy logic irrigation dosing, actuator control, camera driver, edge display, INT8 quantization.

---

## 14. Glossary

| Term | Explanation |
|---|---|
| **EfficientNet-B0** | A neural network architecture that scales depth, width, and resolution together for efficiency. Used for leaf disease classification. |
| **MobileNetV3-Small** | A very lightweight neural network for mobile/edge devices. Used for the leaf/fruit router. |
| **YOLOv8n** | "You Only Look Once" nano — a real-time object detector. Draws bounding boxes around detected objects. Used for fruit disease detection. |
| **Fine-tuning** | Starting from a model pre-trained on a large dataset and continuing training on your specific dataset. Much faster than training from scratch. |
| **Transfer learning** | The general concept: transferring learned knowledge from one task to another via fine-tuning. |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization. Enhances local contrast in images, making disease patterns more visible. |
| **Softmax** | Converts raw model scores into probabilities summing to 1.0. |
| **Argmax** | Picks the index with the highest value — used to select the predicted class. |
| **F1 score** | Harmonic mean of precision and recall. Ranges 0–1. Good for imbalanced class distributions. |
| **Macro-F1** | Average F1 across all classes, treating each class equally regardless of image count. |
| **mAP@0.5** | Mean Average Precision at IoU≥0.5. Standard detection metric — requires both correct class and 50%+ bounding box overlap. |
| **IoU** | Intersection over Union. Measures overlap between predicted and ground-truth boxes. IoU = (overlap area) / (union area). |
| **Precision** | Of all times model predicted class X, how often was it actually X? High precision = few false alarms. |
| **Recall** | Of all actual class X images, how many did the model find? High recall = few misses. |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping. Highlights which image pixels most influenced the model's prediction. Red = important. |
| **Bounding box** | Rectangle around a detected object, defined by [x1, y1, x2, y2]. |
| **NMS** | Non-Maximum Suppression. Removes duplicate overlapping detections, keeping only the highest-confidence one per region. |
| **Checkpoint (.pt)** | A saved snapshot of a model's learned weights. Loading it restores the model to its trained state. |
| **Held-out test set** | Data locked away and never used in training or any decision-making. Used only to measure final real-world performance. |
| **Early stopping** | Automatically halts training when validation performance stops improving, preventing overfitting. |
| **Overfitting** | Model memorises training data and performs poorly on new images. Avoided here by data augmentation, dropout, and early stopping. |
| **Severity percentage** | Estimated fraction of the leaf canopy (or fruit surface) that appears diseased. |
| **Router** | The first model in the pipeline — routes each input image to the correct branch (leaf or fruit). |
| **Gradio** | Python library for building simple browser-based interfaces for ML models. |
| **Group-aware split** | Splitting data so that augmented copies of the same source image always stay in the same partition (train/val/test). |

---

*End of report. For the simplified four-class variant (Early blight, Blossom_End_Rot, Healthy leaf, Healthy plant) see the `4-class-classifier` branch and its `report_version1.md`.*
