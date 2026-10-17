# Checkpoints

The three v1 checkpoints are **tracked in git** (~28 MB total) via explicit
exceptions in `.gitignore`; any other `.pt` file is ignored.

## Files

| Model | Filename |
|---|---|
| Router (MobileNetV3-Small) | `router_v1_mobilenet_v3_small.pt` |
| Leaf classifier (EfficientNet-B0) | `leaf_v1_efficientnet_b0.pt` |
| Fruit detector (YOLOv8n) | `fruit_v1_yolov8n.pt` |

`src/infer_pipeline.py` loads these by default. Use other weights with
`--router-checkpoint`, `--leaf-checkpoint`, `--fruit-checkpoint`.

## Versioning rule

The v1 files are frozen. New weights get a `v2` name (for example
`leaf_v2_efficientnet_b0.pt`): copy the training config and change
`checkpoint.filename`. `src/train.py` and `src/train_fruit_yolo.py` refuse to
overwrite an existing checkpoint unless `--overwrite` is passed. To track a new
checkpoint in git, add a `!checkpoints/<name>.pt` exception to `.gitignore`.

## Verified metrics (v1)

| Model | Key metric |
|---|---|
| Router | 100% val/test accuracy |
| Leaf | Test macro-F1 0.9942 |
| Fruit | Test mAP@0.5 0.807 |

Reports: `results/leaf_classifier_test_report.md`, `results/fruit_detector_test_report.md`.
