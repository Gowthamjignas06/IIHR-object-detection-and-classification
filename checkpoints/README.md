# Checkpoints

Model weights (`.pt` files) are **gitignored** because they are large binaries (~43 MB total).

## Expected filenames

| Model | Filename |
|---|---|
| Router (MobileNetV3-Small) | `router_v1_mobilenet_v3_small.pt` |
| Leaf classifier (EfficientNet-B0) | `leaf_v1_efficientnet_b0.pt` |
| Fruit detector (YOLOv8n) | `fruit_v1_yolov8n.pt` |

Place all three files in this directory before running `src/infer_pipeline.py`.

## Options

1. **Train locally** — see the main [README](../README.md) training section.
2. **GitHub Release** — attach the three `.pt` files to a release and download them into this folder (recommended for portfolio viewers who will not retrain).

## Verified metrics (this build)

| Model | Key metric |
|---|---|
| Router | 100% val/test accuracy |
| Leaf | Test macro-F1 0.9942 |
| Fruit | Test mAP@0.5 0.807 |

Reports: `results/leaf_classifier_test_report.md`, `results/fruit_detector_test_report.md`.
