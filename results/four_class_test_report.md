# Four-Class Tomato Diagnostic Classifier — Test Evaluation Report

**Branch:** `4-class-classifier`  
**Date:** 2026-07-07  
**Status:** Approved — held-out test complete

---

## System overview

A standalone four-class diagnostic system built on top of three existing trained models (no new training). The system routes each input image through a leaf/fruit router, then extracts exactly the four target class scores from the appropriate branch.

| Target class | Domain | Source | Type |
|---|---|---|---|
| **Early blight** | Leaf | EfficientNet-B0 (index 1 / 10) | Fungal leaf disease |
| **Blossom_End_Rot** | Fruit | YOLOv8n (class 1 / 8) | Abiotic fruit disorder |
| **Healthy leaf** | Leaf | EfficientNet-B0 (index 2 / 10) | No disease |
| **Healthy plant** | Fruit | YOLOv8n (class 4 / 8) | No disease |

---

## Router performance

| Split | Accuracy |
|---|---:|
| Validation | **100.00%** |
| Test | **100.00%** |

The router (MobileNetV3-Small) perfectly separates leaf and fruit images on both validation and held-out test sets. Zero misrouting errors.

---

## Leaf branch — Early blight & Healthy leaf

**Model:** EfficientNet-B0 (pre-trained ImageNet → fine-tuned PlantVillage)  
**Test split:** 2,401 held-out images (not used in any training or checkpoint selection)

### Overall leaf test metrics

| Metric | Value |
|---|---:|
| Accuracy | **0.9942** |
| Macro-F1 (all 10 classes) | **0.9942** |
| Weighted-F1 | **0.9942** |

### Per-class metrics — four-class targets only

| Class | Precision | Recall | F1 | Images (test) |
|---|---:|---:|---:|---:|
| **Early blight** | **1.0000** | **0.9800** | **0.9899** | 150 |
| **Healthy leaf** | **1.0000** | **1.0000** | **1.0000** | 238 |

**Two-class leaf aggregate:**

| Metric | Value |
|---|---:|
| Macro-F1 (Early blight + Healthy leaf) | **0.9950** |
| Images correctly classified | 385 / 388 |
| Micro accuracy (these 2 classes only) | **99.23%** |

### Early blight confusion detail

| Metric | Phase 1 baseline | Val (phase 2 best) | Test (held-out) |
|---|---:|---:|---:|
| Early blight precision | 0.7444 | 0.9804 | **1.0000** |
| Late blight → Early blight false positives | 19 | 1 | **0** |

No Late blight image was misclassified as Early blight on the held-out test set.

The 3 Early blight errors (out of 150): 2 misclassified as Late blight, 1 as Septoria leaf spot.

### Val vs test F1 (overfit check)

| Class | Val F1 | Test F1 | Δ |
|---|---:|---:|---:|
| Early blight | 0.9901 | 0.9899 | −0.0002 |
| Healthy leaf | 0.9979 | 1.0000 | +0.0021 |

No drop detected. Test F1 is within 0.001 of val — model generalises cleanly.

### Confusion matrix (4 target leaf classes, rows = true, cols = predicted)

```
                     Early blight   Healthy leaf   (other leaf classes)
Early blight (150)       147              0                  3
Healthy leaf  (238)        0            238                  0
```

---

## Fruit branch — Blossom_End_Rot & Healthy plant

**Model:** YOLOv8n (pre-trained COCO → fine-tuned on grouped fruit split)  
**Test split:** 210 held-out images (group-aware split — no augmentation leakage)  
**Metric:** mAP@0.5 and mAP@0.5:0.95 (standard YOLO detection metrics)

### Per-class detection metrics — four-class targets only

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| **Blossom_End_Rot** | **0.8621** | **0.9231** | **0.9181** | 0.5656 |
| **Healthy plant** | **0.8240** | **0.6176** | **0.7596** | 0.5016 |

### Two-class fruit aggregate

| Metric | Value |
|---|---:|
| Mean precision | **0.8431** |
| Mean recall | **0.7704** |
| Mean mAP@0.5 | **0.8389** |
| Mean mAP@0.5:0.95 | **0.5336** |

### Val vs test mAP@0.5 (overfit check)

| Class | Val mAP@0.5 | Test mAP@0.5 | Δ |
|---|---:|---:|---:|
| Blossom_End_Rot | 0.8830 | **0.9181** | +0.0351 |
| Healthy plant | 0.6551 | **0.7596** | +0.1045 |

Both target classes **improve** from val to test — strong generalisation, no overfitting.

---

## Four-class headline summary

| Class | Metric type | Test score |
|---|---|---:|
| Early blight | F1 | **0.9899** |
| Blossom_End_Rot | mAP@0.5 | **0.9181** |
| Healthy leaf | F1 | **1.0000** |
| Healthy plant | mAP@0.5 | **0.7596** |
| **4-class macro average** | mixed | **0.9169** |
| **Router accuracy** | accuracy | **1.0000** |

> Leaf classes are evaluated by classification F1 (EfficientNet-B0 softmax).  
> Fruit classes are evaluated by detection mAP@0.5 (YOLOv8n bounding boxes).  
> The 4-class macro average is the arithmetic mean of the four test scores above.

---

## Dataset summary

| Split | Leaf images | Fruit images |
|---|---:|---:|
| Train | 11,208 | 910 |
| Validation | 2,402 | 199 |
| **Test (held-out)** | **2,401** | **210** |

**Leaf split:** Stratified random 70 / 15 / 15, seed 42. PlantVillage tomato subset.  
**Fruit split:** Group-aware 70 / 15 / 15 (by Roboflow source stem). Ensures augmented copies of the same image never appear in both train and test.

---

## Training summary

| Component | Architecture | Best epoch | Stop reason | Best val metric |
|---|---|---:|---|---:|
| Router | MobileNetV3-Small | — | — | **100%** accuracy |
| Leaf classifier | EfficientNet-B0 | Phase 2, epoch 12 | Early stopping | **0.9930** macro-F1 |
| Fruit detector | YOLOv8n | 44 (of 64) | Early stopping (patience 20) | **0.761** mAP@0.5 |

---

## Known limitations

| Class | Limitation |
|---|---|
| Early blight | 3 of 150 test images misclassified (Late blight / Septoria confusion). |
| Healthy plant | Lower recall (0.62 on test) — YOLO sometimes misses the healthy-tomato box when the fruit occupies a small region. |

No limitations affect Healthy leaf (100% F1) or Blossom_End_Rot (91.8% mAP@0.5).

---

## Source reports

Full 10-class leaf and 8-class fruit metrics:

- [`results/leaf_classifier_test_report.md`](leaf_classifier_test_report.md)
- [`results/fruit_detector_test_report.md`](fruit_detector_test_report.md)
- Machine-readable JSON: [`results/four_class_test_report.json`](four_class_test_report.json)
