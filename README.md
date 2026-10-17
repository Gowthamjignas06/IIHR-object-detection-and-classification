# Tomato Disease Diagnosis

Dual-branch machine learning pipeline for **tomato leaf disease classification** and **fruit defect detection**, with automatic routing, explainability (Grad-CAM on leaves), severity scoring, and a unified JSON inference API.

Capstone project — software phases 1–2 complete; hardware/sensor integration (phase 3) deferred.

## Results (held-out test)

| Component | Model | Metric |
|---|---|---|
| Router | MobileNetV3-Small | **100%** val/test accuracy |
| Leaf classifier | EfficientNet-B0 | **0.9942** macro-F1 (10 classes) |
| Fruit detector | YOLOv8n | **0.807** mAP@0.5 (8 classes) |

Detailed reports: [`results/leaf_classifier_test_report.md`](results/leaf_classifier_test_report.md) · [`results/fruit_detector_test_report.md`](results/fruit_detector_test_report.md)

## Pipeline

```
Photo → Router (leaf | fruit)
          ├─ Leaf  → EfficientNet-B0 → class + Grad-CAM + severity_pct
          └─ Fruit → YOLOv8n        → boxes + severity_pct
        → Unified JSON
```

Architecture and design rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Quick start

### 1. Clone and install (Python 3.10+)

Windows (PowerShell):

```powershell
git clone https://github.com/Gowthamjignas06/IIHR-object-detection-and-classification.git
cd IIHR-object-detection-and-classification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
git clone https://github.com/Gowthamjignas06/IIHR-object-detection-and-classification.git
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

### 2. Launch the web interface (recommended)

```bash
python src/app.py
```

Opens `http://127.0.0.1:7860` in your browser. Upload any tomato image — leaf or fruit — and click **Diagnose**. Result images and a running `app_log.json` are saved to `results/inference_results/` (gitignored).

### 3. Model weights

The three trained checkpoints are **included in this repo** under [`checkpoints/`](checkpoints/):

| File | Model |
|---|---|
| `router_v1_mobilenet_v3_small.pt` | Leaf vs fruit router |
| `leaf_v1_efficientnet_b0.pt` | 10-class leaf classifier |
| `fruit_v1_yolov8n.pt` | 8-class fruit detector |

No download step is required for inference after cloning.

### 4. Leaf training data (optional for inference only)

Inference works on any photo with the included checkpoints. **Retraining the leaf branch** requires PlantVillage tomato images — see [`data/README.md`](data/README.md).

Fruit split images and labels are included under `data/splits/fruit/`.

### 5. Example: run on held-out test images

These are real images from the **held-out test splits** (not used during training). Ground-truth labels are shown for comparison.

#### Fruit test image (in repo — runs immediately after clone)

**Image:** `data/splits/fruit/images/test/Blossom_end_root_15-Copy_zoom_4_png.rf.8bd26be077f49081bbbe121a5e00408f.jpg`  
**Ground truth:** `Blossom_End_Rot` (fruit test split)

```powershell
python -u src/infer_pipeline.py --json "data/splits/fruit/images/test/Blossom_end_root_15-Copy_zoom_4_png.rf.8bd26be077f49081bbbe121a5e00408f.jpg"
```

**Output:**

```json
{
  "domain": "fruit",
  "class": "Blossom_End_Rot",
  "confidence": 0.693,
  "severity_pct": 49.32,
  "gradcam_overlay": null,
  "detections": [
    {
      "class": "Blossom_End_Rot",
      "confidence": 0.693,
      "bbox": [369.0, 94.5, 525.3, 307.2]
    },
    {
      "class": "Catfaced",
      "confidence": 0.2552,
      "bbox": [0.1, 0.0, 160.1, 289.7]
    }
  ],
  "model_version": "fruit_v1_yolov8n"
}
```

| Field | Value | Notes |
|---|---|---|
| `class` | Blossom_End_Rot | Matches ground truth (highest-confidence box) |
| `confidence` | 0.693 | Primary detection score |
| `severity_pct` | 49.32 | Sum of disease box areas ÷ estimated fruit area |
| `detections` | 2 boxes | YOLO can return multiple defects per image; primary class = top confidence |

#### Leaf test image (PlantVillage — link `data/leaf_raw/` first)

Leaf test images are **not** shipped in this repo (16k images). After linking PlantVillage under `data/leaf_raw/PlantVillage/` (see [`data/README.md`](data/README.md)), run:

**Image:** `data/leaf_raw/PlantVillage/Tomato_Early_blight/07324d80-17c5-4541-9702-62e9c744ac07___RS_Erly.B 7455.JPG`  
**Ground truth:** `Early blight` (leaf test CSV, row 321)

```powershell
python -u src/infer_pipeline.py --json "data/leaf_raw/PlantVillage/Tomato_Early_blight/07324d80-17c5-4541-9702-62e9c744ac07___RS_Erly.B 7455.JPG"
```

**Output:**

```json
{
  "domain": "leaf",
  "class": "Early blight",
  "confidence": 0.9446,
  "severity_pct": 4.42,
  "gradcam_overlay": "results/gradcam_overlays/07324d80-17c5-4541-9702-62e9c744ac07___RS_Erly.B 7455_gradcam.jpg",
  "detections": null,
  "model_version": "leaf_v1_efficientnet_b0"
}
```

| Field | Value | Notes |
|---|---|---|
| `class` | Early blight | Matches ground truth |
| `confidence` | 0.9446 | Softmax probability of predicted class |
| `severity_pct` | 4.42 | HSV/Lab lesion fraction on green canopy |
| `gradcam_overlay` | `.jpg` in `results/gradcam_overlays/` | Written automatically; highlights regions influencing the prediction |

#### Both branches in one command

```powershell
python -u src/infer_pipeline.py --json `
  "data/leaf_raw/PlantVillage/Tomato_Early_blight/07324d80-17c5-4541-9702-62e9c744ac07___RS_Erly.B 7455.JPG" `
  "data/splits/fruit/images/test/Blossom_end_root_15-Copy_zoom_4_png.rf.8bd26be077f49081bbbe121a5e00408f.jpg"
```

The router sends each image to the correct branch automatically (leaf → classifier + Grad-CAM; fruit → YOLO boxes).

### 6. Run inference on your own photo

From the repo root:

```powershell
python -u src/infer_pipeline.py --json "path\to\your_photo.jpg"
```

Multiple images:

```powershell
python -u src/infer_pipeline.py --json photo1.jpg photo2.jpg
```

**CLI flags**

| Flag | Description |
|---|---|
| `IMAGE ...` | One or more image paths (required) |
| `--json` | Print JSON only (no path headers) |
| `--device` | `auto` (default), `cuda`, or `cpu` |
| `--router-checkpoint`, `--leaf-checkpoint`, `--fruit-checkpoint` | Use other weights (e.g. a v2 checkpoint); `model_version` in the JSON is the checkpoint file name |

Relative image paths are resolved against your current directory first, then the repo root.

Leaf runs save a Grad-CAM overlay to `results/gradcam_overlays/<stem>_gradcam.jpg`.

### Disease guide: explanation, treatment plan, action steps

Every result also carries an additive `advice` key (the original seven keys are unchanged), shown as a **Disease guide** in the web app:

| Field | Contents |
|---|---|
| `condition`, `category`, `contagious` | Plain-language name; fungal / water-mould / bacterial / virus / pest / physiological |
| `explanation`, `cause`, `symptoms`, `why_it_matters` | What the condition is and how to recognise it |
| `urgency`, `urgency_reason` | `none` / `low` / `moderate` / `high` — scales with `severity_pct`, always `high` for viruses and late blight, capped at `moderate` for non-infectious disorders |
| `treatment_plan` | `cultural`, `organic`, `chemical` (active ingredients only — use locally registered products) |
| `action_steps` | `immediate`, `this_week`, `prevention` |
| `notes`, `get_help`, `disclaimer` | Low-confidence / secondary-detection notes, IIHR / KVK referral, safety disclaimer |

Content lives in [`knowledge/disease_advice.yaml`](knowledge/disease_advice.yaml) (one entry per model class) and is marked **draft — not yet reviewed by an agronomist**. Use `--no-advice` to omit the key.

### Phone photos — practical notes

- Convert **HEIC → JPEG** before running (Pillow does not open `.heic` by default).
- EXIF orientation is applied automatically, so phone photos are processed upright.
- Crop so the **leaf or fruit fills most of the frame** for best routing and severity estimates.

## Training

```powershell
# Leaf manifests (after data/leaf_raw is set up)
python src/split_leaf_manifest.py --write
python src/build_router_manifest.py --write

# Router
python src/train.py --config configs/router.yaml

# Leaf classifier (two-phase fine-tune in config)
python src/train.py --config configs/leaf.yaml

# Fruit YOLO
python src/train_fruit_yolo.py --config configs/fruit_yolo.yaml
```

Training refuses to overwrite an existing checkpoint (the v1 files are frozen): copy the config, set `checkpoint.filename` to a v2 name, or pass `--overwrite`.

Evaluation scripts: `scripts/eval_leaf_test_report.py --checkpoint …`, `scripts/eval_fruit_test_report.py --checkpoint … [--out …]` (writes a new report and compares it with v1; never overwrites the v1 report), `scripts/verify_leaf_checkpoint.py`

Tests: `python -m pytest tests/`

## Project layout

```
├── configs/          # YAML training configs
├── data/             # Splits, audit, data setup docs
├── checkpoints/      # Trained model weights (.pt)
├── results/          # Test reports + Grad-CAM outputs
├── scripts/          # Evaluation and analysis utilities
└── src/              # Training, inference, severity, Grad-CAM
```

## Known limitations

- **Catfaced** (fruit): high recall, moderate precision; confusion with Fruit_Cracking.
- **Spotted_Wilt_Virus** (fruit): recall limited by data scarcity and Late_Blight visual overlap.
- **Fruit severity_pct**: v1 box-geometry heuristic; sensitive to photo cropping (see fruit test report).

## License

Code: [MIT](LICENSE). Training datasets remain under their original licenses (PlantVillage, Kaggle/Roboflow fruit export).

## Citation / datasets

- Leaf: PlantVillage (`emmarex/plantdisease` on Kaggle)
- Fruit: `profnourasemary/tomato-dataset` (Roboflow YOLOv8 export)
