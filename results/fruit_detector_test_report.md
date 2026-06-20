# Fruit YOLO Detector — Held-Out Test Evaluation Report

**Step 6 — approved complete**  
**Date:** 2026-06-20  
**Checkpoint:** `checkpoints/fruit_v1_yolov8n.pt`

## Checkpoint summary

| Field | Value |
|---|---|
| Model | YOLOv8n |
| Best epoch | 44 |
| Stop reason | Early stopping (patience=20) at epoch 64 |
| Canonical artifact | `checkpoints/fruit_v1_yolov8n.pt` |

## Test-set overall metrics (held-out, n=210 images)

| Metric | Val | Test |
|---|---:|---:|
| mAP@0.5 | 0.7611 | **0.8067** |
| mAP@0.5:0.95 | 0.4455 | **0.4714** |
| Mean precision | 0.8256 | 0.8105 |
| Mean recall | 0.6410 | 0.7402 |

## Per-class metrics (test)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Anthracnose | 0.7790 | 0.9234 | 0.8507 | 0.4752 |
| Blossom_End_Rot | 0.8621 | 0.9231 | 0.9181 | 0.5656 |
| Catfaced | 0.7849 | 0.8500 | 0.8300 | 0.3938 |
| Fruit_Cracking | 0.6984 | 0.5758 | 0.6888 | 0.4502 |
| Healthy_Tomato | 0.8240 | 0.6176 | 0.7596 | 0.5016 |
| Late_Blight | 0.8754 | 0.7619 | 0.8646 | 0.5379 |
| Mold | 0.7777 | 0.5614 | 0.6884 | 0.3333 |
| Spotted_Wilt_Virus | 0.8821 | 0.7083 | 0.8533 | 0.5140 |

## Val vs test mAP (overfit check, threshold −0.05)

| Class | Val mAP@0.5 | Test mAP@0.5 | Δ | Flag |
|---|---:|---:|---:|---|
| Anthracnose | 0.8180 | 0.8507 | +0.0327 | — |
| Blossom_End_Rot | 0.8830 | 0.9181 | +0.0351 | — |
| Catfaced | 0.8913 | 0.8300 | -0.0613 | SEVERE |
| Fruit_Cracking | 0.7496 | 0.6888 | -0.0608 | SEVERE |
| Healthy_Tomato | 0.6551 | 0.7596 | +0.1045 | — |
| Late_Blight | 0.7474 | 0.8646 | +0.1172 | — |
| Mold | 0.6634 | 0.6884 | +0.0250 | — |
| Spotted_Wilt_Virus | 0.6809 | 0.8533 | +0.1724 | — |

## Known limitations (this iteration)

### Catfaced
- Recall-perfect but precision-poor (over-detection)
- Val precision 0.568, recall 1.0
- Dominant FP: **Fruit_Cracking -> pred Catfaced (40% of FP)**

### Spotted_Wilt_Virus
- Low recall; class confusion with Late_Blight at high IoU
- Val recall 0.368, precision 0.886
- 31.6% of GT missed as background; 26.3% absorbed as Late_Blight (79–89% IoU)
- **Conclusion:** More labeled data needed, not more training epochs on this split.

## Inference severity note (v1 heuristic)

Fruit `severity_pct` in `infer_pipeline.py` uses box geometry (`src/severity.py`): sum of disease box areas divided by an estimated fruit area (healthy-box union, multi-box union, or **full image area** when only one disease detection exists). This is a **v1 heuristic**, not a calibrated fruit-surface measurement — it is sensitive to how tightly each photo is cropped to the fruit.

## Related artifacts

- Training log: `logs/fruit_train.log`
- Confusion analysis: `scripts/analyze_fruit_confusion.py`
- Machine-readable report: `fruit_detector_test_report.json`
