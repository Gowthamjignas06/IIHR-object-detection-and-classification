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

### 1. Clone and install

```powershell
git clone https://github.com/Gowthamjignas06/IIHR-object-detection-and-classification.git
cd IIHR-object-detection-and-classification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch with CUDA if you have an NVIDIA GPU ([pytorch.org](https://pytorch.org)):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 2. Model weights

Weights are not in git (~43 MB). Either:

- **Train** (see [Training](#training)), or  
- **Download** the three `.pt` files from a GitHub Release into [`checkpoints/`](checkpoints/)

Required files: `router_v1_mobilenet_v3_small.pt`, `leaf_v1_efficientnet_b0.pt`, `fruit_v1_yolov8n.pt`

### 3. Leaf training data (optional for inference only)

Inference works on any photo if checkpoints are present. **Training the leaf branch** requires PlantVillage tomato images — see [`data/README.md`](data/README.md).

Fruit split images and labels are included under `data/splits/fruit/`.

### 4. Run inference

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

Leaf runs save a Grad-CAM overlay to `results/gradcam_overlays/<stem>_gradcam.jpg`.

**Example output**

```json
{
  "domain": "leaf",
  "class": "Early blight",
  "confidence": 0.9446,
  "severity_pct": 4.42,
  "gradcam_overlay": ".../results/gradcam_overlays/..._gradcam.jpg",
  "detections": null,
  "model_version": "leaf_v1_efficientnet_b0"
}
```

### Phone photos — practical notes

- Convert **HEIC → JPEG** before running (Pillow does not open `.heic` by default).
- **EXIF orientation is not applied** — rotate/export so the image looks correct on disk.
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

Evaluation scripts: `scripts/eval_leaf_test_report.py`, `scripts/eval_fruit_test_report.py`, `scripts/verify_leaf_checkpoint.py`

## Project layout

```
├── configs/          # YAML training configs
├── data/             # Splits, audit, data setup docs
├── checkpoints/      # Model weights (.pt, gitignored)
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
