# Leaf Classifier — Held-Out Test Evaluation Report

**Step 5 — approved complete**  
**Date:** 2026-06-20  
**Checkpoint:** `checkpoints/leaf_v1_efficientnet_b0.pt` (canonical artifact)

## Checkpoint summary

| Field | Value |
|---|---|
| Best phase / epoch | Phase 2, epoch 12 |
| Best val macro-F1 | 0.992957 (0.9930 rounded) |
| Early blight precision (val) | 0.9804 |
| Late blight → Early blight FP (val) | 1 |
| Completed phases | [1, 2] |
| Integrity check | **PASS** — live re-eval delta 0.000000 |

This checkpoint is the **canonical leaf classifier artifact** going forward. No further leaf training is planned unless a downstream issue surfaces.

## Test-set overall metrics (held-out, n=2,401)

| Metric | Test |
|---|---:|
| Accuracy | 0.9942 |
| Macro-F1 | 0.9942 |
| Weighted-F1 | 0.9942 |

Val best macro-F1: **0.9930** — test is **above** val, arguing against val-overfitting from early stopping / checkpoint selection.

## Per-class metrics (test)

| Class | Precision | Recall | F1 | N |
|---|---:|---:|---:|---:|
| Bacterial spot | 0.9969 | 0.9969 | 0.9969 | 319 |
| Early blight | 1.0000 | 0.9800 | 0.9899 | 150 |
| Healthy | 1.0000 | 1.0000 | 1.0000 | 238 |
| Late blight | 0.9930 | 0.9930 | 0.9930 | 287 |
| Leaf Mold | 0.9930 | 0.9930 | 0.9930 | 143 |
| Septoria leaf spot | 0.9925 | 1.0000 | 0.9962 | 265 |
| Spider mites (Two-spotted) | 0.9804 | 0.9921 | 0.9862 | 252 |
| Target Spot | 0.9905 | 0.9905 | 0.9905 | 210 |
| Tomato Yellow Leaf Curl Virus | 0.9979 | 0.9938 | 0.9958 | 481 |
| Tomato mosaic virus | 1.0000 | 1.0000 | 1.0000 | 56 |

## Confusion matrix (test, rows=true × cols=pred)

Class order: Bacterial spot, Early blight, Healthy, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, TYLCV, Tomato mosaic virus.

```
[318, 0, 0, 0, 0, 0, 0, 0, 1, 0]
[0, 147, 0, 2, 0, 1, 0, 0, 0, 0]
[0, 0, 238, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 285, 1, 1, 0, 0, 0, 0]
[0, 0, 0, 0, 142, 0, 1, 0, 0, 0]
[0, 0, 0, 0, 0, 265, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 250, 2, 0, 0]
[0, 0, 0, 0, 0, 0, 2, 208, 0, 0]
[1, 0, 0, 0, 0, 0, 2, 0, 478, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 56]
```

## Early blight & Late blight → Early blight FP

| Metric | Phase 1 baseline | Val (phase 2 best) | Test |
|---|---:|---:|---:|
| Early blight precision | 0.7444 | 0.9804 | **1.0000** |
| Late blight → Early blight FP | 19 | 1 | **0** |

## Val vs test per-class F1 (overfit check)

Threshold: flag if test drops >0.03 below val on F1, precision, or recall.

| Class | Val F1 | Test F1 | Δ (test−val) | Flag |
|---|---:|---:|---:|---|
| Bacterial spot | 0.9969 | 0.9969 | −0.0000 | — |
| Early blight | 0.9901 | 0.9899 | −0.0002 | — |
| Healthy | 0.9979 | 1.0000 | +0.0021 | — |
| Late blight | 0.9930 | 0.9930 | +0.0000 | — |
| Leaf Mold | 0.9965 | 0.9930 | −0.0035 | — |
| Septoria leaf spot | 0.9962 | 0.9962 | +0.0000 | — |
| Spider mites (Two-spotted) | 0.9901 | 0.9862 | −0.0039 | — |
| Target Spot | 0.9809 | 0.9905 | +0.0096 | — |
| Tomato Yellow Leaf Curl Virus | 0.9969 | 0.9958 | −0.0010 | — |
| Tomato mosaic virus | 0.9912 | 1.0000 | +0.0088 | — |

**No class flagged.** Largest F1 dips: Spider mites (−0.0039), Leaf Mold (−0.0035).

## Phase 2 training outcome

- Epochs run: 12 / 15 (early stopping)
- Premature early-stop flag: **not raised**
- Stop reason: val macro-F1 did not improve by >0.003 for 4 consecutive epochs after epoch 12 peak

## Related artifacts

- Training log: `logs/leaf_train_phase2.log`
- Eval log: `logs/leaf_eval_test.log`
- Machine-readable report: `results/leaf_classifier_test_report.json`
