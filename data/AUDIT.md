# Data Audit Report (Build Order Step 1)

Generated as part of the tomato disease diagnosis system build. Source of truth for assumptions: `ARCHITECTURE.md`.

## Data locations

| Branch | Junction path | Actual data root |
|---|---|---|
| Leaf | `data/leaf_raw/` | `plant_village_dataset_only_tomato/PlantVillage/` |
| Fruit | `data/fruit_raw/` | `tomato_disease_fruit/` (Roboflow YOLOv8 export) |

---

## Leaf dataset (PlantVillage tomato subset)

### Summary

- **10 classes** — all expected PlantVillage tomato classes are present
- **16,011 images** total (`.jpg` / `.jpeg`)
- **Layout flag:** class folders live under an extra `PlantVillage/` subdirectory, not directly under `leaf_raw/`. Downstream code should resolve `leaf_raw/PlantVillage/<class>/`.

### Class folder mapping

Architecture human-readable names vs local folder names:

| Architecture label | Local folder | Image count |
|---|---|---:|
| Bacterial spot | `Tomato_Bacterial_spot` | 2,127 |
| Early blight | `Tomato_Early_blight` | 1,000 |
| Late blight | `Tomato_Late_blight` | 1,909 |
| Leaf Mold | `Tomato_Leaf_Mold` | 952 |
| Septoria leaf spot | `Tomato_Septoria_leaf_spot` | 1,771 |
| Spider mites (Two-spotted) | `Tomato_Spider_mites_Two_spotted_spider_mite` | 1,676 |
| Target Spot | `Tomato__Target_Spot` | 1,404 |
| Tomato Yellow Leaf Curl Virus | `Tomato__Tomato_YellowLeaf__Curl_Virus` | 3,208 |
| Tomato mosaic virus | `Tomato__Tomato_mosaic_virus` | 373 |
| Healthy | `Tomato_healthy` | 1,591 |

### Class imbalance (leaf)

Notable spread: TYLCV (3,208) vs Leaf Mold (952) vs Tomato mosaic virus (373). Weighted loss or weighted sampling will be needed at training time (architecture design principle 5).

### File format

- Almost exclusively `.jpg`; one `.jpeg` in `Tomato_Late_blight`
- Plain folder-per-class layout (image classification, not detection)

---

## Fruit dataset (Roboflow YOLOv8 export)

### Summary

- **1,319 images** total across Roboflow's pre-export splits
- **YOLOv8 object detection** format (bounding boxes, not folder-per-class)
- **8 classes** confirmed from `data.yaml` (see below)
- Images pre-resized to **640×640** (stretch) per `README.roboflow.txt`
- Roboflow applied **3× augmentation** to many (not all) source images

### Roboflow pre-export split (do not use for training)

| Split | Images | Labels |
|---|---:|---:|
| train | 1,155 | 1,155 |
| valid | 110 | 110 |
| test | 54 | 54 |

These splits must **not** be used as-is for model training/evaluation. See grouping analysis below.

### Class list (`data.yaml`)

```yaml
nc: 8
names: ['Anthracnose', 'Blossom_End_Rot', 'Catfaced', 'Fruit_Cracking',
        'Healthy_Tomato', 'Late_Blight', 'Mold', 'Spotted_Wilt_Virus']
```

### Per-class counts (all 1,319 label files)

| Class ID | Class name | Images with ≥1 box | Box instances |
|---:|---|---:|---:|
| 0 | Anthracnose | 202 | 270 |
| 1 | Blossom_End_Rot | 142 | 321 |
| 2 | Catfaced | 83 | 131 |
| 3 | Fruit_Cracking | 204 | 244 |
| 4 | Healthy_Tomato | 274 | 1,039 |
| 5 | Late_Blight | 161 | 309 |
| 6 | Mold | 232 | 304 |
| 7 | Spotted_Wilt_Virus | 54 | 100 |

`Healthy_Tomato` has the most images but also the highest instance count per image (multiple boxes per fruit image is common).

### YOLOv8 label format (confirmed)

Each `.txt` label file has one line per bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]`. Sample lines from `valid/labels/`:

```text
0 0.440625 0.6109375 0.4640625 0.30078125
2 0.54453125 0.49921875 0.84296875 0.99765625
```

Full scan of all 1,319 label files: **0 malformed lines** (every non-empty line has exactly 5 fields).

### Filename convention and source-image grouping

Roboflow export filenames follow:

```
{base_stem}.rf.{32_hex_hash}.{ext}
```

Example triplet (same source, 3 augmented copies):

```
anthracnose10-Copy_jpg.rf.0666efb79a361dcd7a0c8da58c4f77d2.jpg
anthracnose10-Copy_jpg.rf.09425fd86fadc3932ea7bbd732bd13d3.jpg
anthracnose10-Copy_jpg.rf.cc01e86fe8a272e956674cc9f6dc638e.jpg
```

**Grouping key:** filename stem before `.rf.` (e.g. `anthracnose10-Copy_jpg`).

### Grouping statistics (pooled train + valid + test)

| Metric | Value |
|---|---:|
| Total images | 1,319 |
| Unique source groups | **549** |
| Groups with 3 augmented copies | 385 |
| Groups with 1 image only | 164 |
| Cross-split group leakage (Roboflow split) | **0** |

Roboflow split structure (important):

| Roboflow split | Groups | Images | Augmentation pattern |
|---|---:|---:|---|
| train | 385 | 1,155 | All groups are triplets (×3) |
| valid | 110 | 110 | All groups are singletons (×1) |
| test | 54 | 54 | All groups are singletons (×1) |

Implications:

1. No augmented copy of the same source crosses Roboflow's train/valid/test boundaries.
2. However, **train contains all augmented triplets while valid/test contain none** — using Roboflow's split would structurally bias train (more variation) vs val/test.
3. A fresh **group-level** split (step 2) is still required.

### Source groups by primary class (for stratified splitting)

Primary class = mode of class IDs in the group's label file(s):

| Class | Source groups |
|---|---:|
| Healthy_Tomato | 102 |
| Mold | 95 |
| Anthracnose | 83 |
| Fruit_Cracking | 82 |
| Blossom_End_Rot | 66 |
| Late_Blight | 64 |
| Catfaced | 35 |
| Spotted_Wilt_Virus | 22 |

Primary class = mode of class IDs across all label files in the group (used by `split_fruit_groups.py` for stratification).

---

## Architecture mismatches (surfaced explicitly)

| ARCHITECTURE.md assumption | Actual finding |
|---|---|
| ~440 unique fruit source images | **549** unique source groups |
| Fruit class count "confirm from data.yaml" | **8 classes confirmed** (names above) |
| Leaf data directly under `leaf_raw/` | Extra **`PlantVillage/`** nesting layer |
| Roboflow 3× augmentation per source | Applies to **385/549** groups; **164/549** have only 1 copy |
| Fruit is YOLOv8 detection | **Confirmed** — bounding box labels, not classification folders |

Do not silently work around these; downstream configs and docs should reflect the actual counts and paths.

---

## Next step

Run `src/split_fruit_groups.py --dry-run` to preview group-based train/val/test sizes before materializing `data/splits/fruit/`.
