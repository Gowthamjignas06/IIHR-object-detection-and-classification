# Four-Class Tomato Diagnostic Classifier

**Branch:** `4-class-classifier`

A standalone diagnostic system that classifies tomato images into exactly **four conditions** — covering both leaf disease and fruit disorder — using existing trained models without any new training.

| Class | Domain | Type |
|---|---|---|
| **Early blight** | Leaf | Fungal disease (Alternaria solani) |
| **Blossom_End_Rot** | Fruit | Abiotic disorder (calcium deficiency) |
| **Healthy leaf** | Leaf | No disease |
| **Healthy plant** | Fruit | No disorder |

---

## Test results (held-out)

| Class | Metric | Test score |
|---|---|---:|
| Early blight | F1 | **0.9899** |
| Blossom_End_Rot | mAP@0.5 | **0.9181** |
| Healthy leaf | F1 | **1.0000** |
| Healthy plant | mAP@0.5 | **0.7596** |
| **4-class macro average** | — | **0.9169** |
| **Router accuracy** | — | **1.0000** |

Full report: [`results/four_class_test_report.md`](results/four_class_test_report.md)  
Machine-readable JSON: [`results/four_class_test_report.json`](results/four_class_test_report.json)

---

## How it works

The system routes each image through three existing trained models and extracts only the four relevant scores — no new model weights were created.

```
Input image
     │
     ▼
 Router (MobileNetV3-Small)           ← 100% routing accuracy
     │
     ├─ leaf  →  EfficientNet-B0      ← trained on 10 leaf classes
     │             extracts:
     │             • Early blight probability   (class index 1)
     │             • Healthy leaf probability   (class index 2)
     │             → outputs whichever is higher
     │
     └─ fruit →  YOLOv8n              ← trained on 8 fruit classes
                   extracts:
                   • Blossom_End_Rot confidence  (YOLO class 1)
                   • Healthy plant confidence    (YOLO class 4)
                   → outputs whichever is higher
                   (no detection = Healthy plant by default)
```

---

## Quick start

### 1. Clone and install

```powershell
git clone -b 4-class-classifier https://github.com/Gowthamjignas06/IIHR-object-detection-and-classification.git
cd IIHR-object-detection-and-classification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
git clone -b 4-class-classifier https://github.com/Gowthamjignas06/IIHR-object-detection-and-classification.git
cd IIHR-object-detection-and-classification
python3 -m venv .venv          # macOS without 3.10+: brew install python@3.12, then python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` pins the exact versions verified on macOS arm64 (CPU, Python 3.12).

Install PyTorch with CUDA if you have an NVIDIA GPU ([pytorch.org](https://pytorch.org)):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 2. Model weights

The three trained checkpoints are tracked in git under `checkpoints/` (use other weights with `--router-checkpoint`, `--leaf-checkpoint`, `--fruit-checkpoint`):

| File | Architecture | Role |
|---|---|---|
| `router_v1_mobilenet_v3_small.pt` | MobileNetV3-Small | Leaf / fruit router |
| `leaf_v1_efficientnet_b0.pt` | EfficientNet-B0 | Leaf disease classifier (10 classes) |
| `fruit_v1_yolov8n.pt` | YOLOv8n | Fruit defect detector (8 classes) |

These are the same checkpoints trained in the `main` branch — no new training is required.

### 3. Launch the web interface (recommended)

```powershell
python src/app.py
```

This opens a browser window at `http://127.0.0.1:7860`. Drag-and-drop or upload any tomato image and click **Classify**.

Every run automatically saves:
- An **annotated result image** (`results/inference_results/YYYYMMDD_HHMMSS_<class>_result.jpg`)
- A **JSON log entry** appended to `results/inference_results/app_log.json` (outputs are gitignored)

Below the result, a **Disease guide** explains the condition and gives a treatment plan and action steps (see [Disease guide](#disease-guide-explanation-treatment-plan-action-steps)).

### 4. Run inference from the command line

**Single image:**

```powershell
python src/four_class_infer.py path\to\tomato_image.jpg
```

**Multiple images:**

```powershell
python src/four_class_infer.py img1.jpg img2.jpg img3.jpg
```

**JSON output only (for programmatic use):**

```powershell
python src/four_class_infer.py path\to\image.jpg --json
```

---

## Example outputs

### Early blight leaf

```powershell
python src/four_class_infer.py "data/leaf_raw/PlantVillage/Tomato_Early_blight/07324d80-17c5-4541-9702-62e9c744ac07___RS_Erly.B 7455.JPG"
```

```
============================================================
Image : ...Tomato_Early_blight/07324d80...JPG
Domain: leaf  (router confidence: 0.9987)

  Predicted class : Early blight
  Confidence      : 0.9446

  Four-class scores:
    Early blight          0.9446 <-- predicted
    Blossom_End_Rot       0.0000
    Healthy leaf          0.0218
    Healthy plant         0.0000

  Full leaf model top-3:
    Early blight                        0.9446
    Late blight                         0.0312
    Target Spot                         0.0109
```

### Blossom_End_Rot fruit

```powershell
python src/four_class_infer.py "data/splits/fruit/images/test/Blossom_end_root_15-Copy_zoom_4_png.rf.8bd26be077f49081bbbe121a5e00408f.jpg"
```

```
============================================================
Image : ...Blossom_end_root_15-Copy_zoom_4_png...jpg
Domain: fruit  (router confidence: 0.9963)

  Predicted class : Blossom_End_Rot
  Confidence      : 0.6930

  Four-class scores:
    Early blight          0.0000
    Blossom_End_Rot       0.6930 <-- predicted
    Healthy leaf          0.0000
    Healthy plant         0.0000

  YOLO detections (2 box(es)):
    Blossom_End_Rot           conf=0.6930  bbox=[369.0, 94.5, 525.3, 307.2]
    Catfaced                  conf=0.2552  bbox=[0.1, 0.0, 160.1, 289.7]
```

> Note: YOLO may detect other fruit classes (Catfaced, Mold, etc.) in the same image. The four-class system only surfaces the scores for Blossom_End_Rot and Healthy plant. Any other detection is ignored.

### JSON output format

```json
{
  "predicted_class": "Early blight",
  "confidence": 0.9446,
  "domain": "leaf",
  "router_confidence": 0.9987,
  "class_scores": {
    "Early blight": 0.9446,
    "Blossom_End_Rot": 0.0,
    "Healthy leaf": 0.0218,
    "Healthy plant": 0.0
  },
  "detections": null,
  "raw_leaf_top3": [
    {"class": "Early blight", "score": 0.9446},
    {"class": "Late blight", "score": 0.0312},
    {"class": "Target Spot", "score": 0.0109}
  ],
  "model_versions": {
    "router": "router_v1_mobilenet_v3_small",
    "leaf": "leaf_v1_efficientnet_b0",
    "fruit": "fruit_v1_yolov8n"
  },
  "severity_pct": 4.42,
  "warnings": [],
  "other_conditions": [],
  "advice": {"condition": "Early blight (leaf)", "urgency": "low", "...": "..."}
}
```

The first eight keys are the original format and keep their values. Added keys:

| Key | Meaning |
|---|---|
| `severity_pct` | Affected area % (leaf colour segmentation / fruit box geometry, same as `main`) |
| `warnings` | `no_fruit_detected` — YOLO found nothing (output still says `Healthy plant` 1.0); `other_condition_detected` — the fruit shows only non-target conditions (e.g. Catfaced) but the four-class fallback says `Healthy plant`; `other_leaf_condition_detected` — the 10-class leaf model's top class is neither Early blight nor Healthy |
| `other_conditions` | `[{class, confidence}]` behind those warnings |
| `advice` | Disease guide for what was actually seen (below) |

## Disease guide: explanation, treatment plan, action steps

The `advice` key (and the **Disease guide** section in the app) gives `explanation`, `cause`, `symptoms`, `why_it_matters`, an `urgency` level that scales with `severity_pct` (always high for viruses and late blight; capped at moderate for non-infectious disorders), a `treatment_plan` (cultural / organic / chemical by active ingredient) and `action_steps` (now / this week / prevention), plus an IIHR/KVK referral and a disclaimer. When `other_conditions` is set, the guide covers that condition instead of the four-class fallback. Content: [`knowledge/disease_advice.yaml`](knowledge/disease_advice.yaml) (shared with `main`; draft, not yet reviewed by an agronomist). Use `--no-advice` to omit it.

---

## Python API

```python
from src.four_class_infer import FourClassDiagnosticClassifier

clf = FourClassDiagnosticClassifier(device="auto")

result = clf.classify("path/to/image.jpg")

print(result["predicted_class"])   # e.g. "Early blight"
print(result["confidence"])        # e.g. 0.9446
print(result["class_scores"])      # scores for all 4 classes
```

**Constructor options:**

| Parameter | Default | Description |
|---|---|---|
| `device` | `"auto"` | `auto`, `cpu`, or `cuda` |
| `fruit_conf_threshold` | `0.20` | YOLO box confidence floor |
| `fruit_iou_threshold` | `0.70` | YOLO NMS IoU threshold |

---

## Detailed test results

### Dataset

| Split | Leaf images | Fruit images |
|---|---:|---:|
| Train | 11,208 | 910 |
| Validation | 2,402 | 199 |
| **Test (held-out)** | **2,401** | **210** |

- **Leaf:** Stratified 70 / 15 / 15 split, seed 42. Source: PlantVillage tomato subset (Kaggle).
- **Fruit:** Group-aware 70 / 15 / 15 split. Source: Roboflow YOLOv8 export. Augmented copies of the same image are never split across train and test.

### Early blight (leaf) — classification metrics

| Metric | Validation | Test (held-out) |
|---|---:|---:|
| Precision | 0.9804 | **1.0000** |
| Recall | 0.9800 | **0.9800** |
| F1 | 0.9901 | **0.9899** |
| Support | 240 | 150 |

- **Zero** Late blight images misclassified as Early blight on the test set (vs 19 in the phase 1 baseline).
- 3 test errors: 2 Early blight → Late blight, 1 Early blight → Septoria leaf spot.

### Healthy leaf — classification metrics

| Metric | Validation | Test (held-out) |
|---|---:|---:|
| Precision | 0.9979 | **1.0000** |
| Recall | 0.9979 | **1.0000** |
| F1 | 0.9979 | **1.0000** |
| Support | 476 | 238 |

Perfect recall and precision on the held-out test set.

### Blossom_End_Rot (fruit) — detection metrics

| Metric | Validation | Test (held-out) |
|---|---:|---:|
| Precision | 0.8251 | **0.8621** |
| Recall | 0.8298 | **0.9231** |
| mAP@0.5 | 0.8830 | **0.9181** |
| mAP@0.5:0.95 | 0.4994 | **0.5656** |

Test mAP@0.5 exceeds validation — the model generalises well on this class.

### Healthy plant (fruit) — detection metrics

| Metric | Validation | Test (held-out) |
|---|---:|---:|
| Precision | 0.8355 | **0.8240** |
| Recall | 0.4286 | **0.6176** |
| mAP@0.5 | 0.6551 | **0.7596** |
| mAP@0.5:0.95 | 0.3586 | **0.5016** |

Test mAP@0.5 substantially exceeds validation (YOLO detects healthy tomatoes more reliably on the test set).

### Training details

| Component | Architecture | Best epoch | Stop reason | Best val metric |
|---|---|---:|---|---:|
| Router | MobileNetV3-Small | — | — | 100% accuracy |
| Leaf classifier | EfficientNet-B0 | Phase 2, epoch 12 / 15 | Early stopping | 0.9930 macro-F1 |
| Fruit detector | YOLOv8n | 44 / 64 | Early stopping (patience 20) | 0.761 mAP@0.5 |

---

## Practical notes

- **Image format:** JPEG, PNG. Convert HEIC before running. EXIF orientation is applied automatically.
- **Framing:** Crop so the leaf or fruit fills most of the frame for best routing.
- **Multiple fruit defects:** YOLO may detect several boxes in one image. Only the Blossom_End_Rot and Healthy plant boxes influence the four-class output; any other detections are informational only (visible in `detections` field).
- **Tests:** `python -m pytest tests/` (four-class and unified JSON contracts, knowledge base coverage).
- **GPU:** Inference runs on CPU by default. Pass `--device cuda` or set `device="cuda"` for faster batch processing.

---

## Known limitations

| Class | Limitation |
|---|---|
| Early blight | 3 / 150 test images misclassified (Late blight or Septoria visual similarity). |
| Healthy plant | Recall = 0.62 on test — YOLO occasionally misses a healthy-tomato box when the fruit occupies a small area of the frame. |
| Out-of-scope input | If an image shows a different leaf disease (e.g. Bacterial spot), `predicted_class` is still "Early blight" or "Healthy leaf" (and a fruit with only other defects is "Healthy plant" 1.0). The added `warnings` / `other_conditions` keys flag these cases; there is still no "unknown" class. Non-tomato photos are routed to leaf/fruit with high router confidence (see main's external validation report). |

---

## Project layout

```
├── src/
│   └── four_class_infer.py   ← four-class inference module (this branch)
│   └── infer_pipeline.py     ← original 10+8 class pipeline (main)
├── configs/
│   └── four_class.yaml       ← class mapping config (this branch)
├── checkpoints/              ← trained model weights (.pt)
├── results/
│   └── four_class_test_report.md    ← detailed test report (this branch)
│   └── four_class_test_report.json  ← machine-readable metrics
│   └── leaf_classifier_test_report.md
│   └── fruit_detector_test_report.md
└── data/splits/              ← train / val / test manifests
```

---

## License

Code: [MIT](LICENSE). Training datasets remain under their original licenses (PlantVillage: academic non-commercial; Roboflow fruit export: see source).

## Datasets

- **Leaf:** PlantVillage tomato subset — `emmarex/plantdisease` on Kaggle
- **Fruit:** `profnourasemary/tomato-dataset` on Roboflow (YOLOv8 export)
