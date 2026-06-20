# Data setup

This repo ships **split manifests, fruit labels, and fruit split images** (~60 MB). Raw leaf images and the Roboflow export are **not** committed (size + licensing). You must link or download them locally before training the leaf branch or regenerating fruit splits.

Full audit details: [`AUDIT.md`](AUDIT.md).

## Layout

| Path | Purpose |
|---|---|
| `data/leaf_raw/` | PlantVillage tomato subset (16,011 images) |
| `data/fruit_raw/` | Roboflow YOLOv8 export (optional if using committed fruit split) |
| `data/splits/leaf/` | Train/val/test CSV manifests |
| `data/splits/fruit/` | Group-split YOLO layout (`images/`, `labels/`, `data.yaml`) |

## Leaf data (PlantVillage)

1. Download the PlantVillage dataset (Kaggle: `emmarex/plantdisease`) or use a pre-filtered tomato-only subset.
2. Place class folders so images live under:
   ```
   data/leaf_raw/PlantVillage/Tomato_Early_blight/
   data/leaf_raw/PlantVillage/Tomato_healthy/
   ...
   ```
3. On Windows you can use a directory junction instead of copying:
   ```powershell
   cmd /c mklink /J data\leaf_raw\PlantVillage "C:\path\to\plant_village\PlantVillage"
   ```
4. Generate manifests:
   ```powershell
   python src/split_leaf_manifest.py --write
   python src/build_router_manifest.py --write
   ```

## Fruit data (Roboflow / Kaggle)

The committed `data/splits/fruit/` folder is enough to **train and evaluate the fruit detector** without re-downloading.

To regenerate splits from the raw Roboflow export:

1. Download `profnourasemary/tomato-dataset` (Kaggle) or the Roboflow export.
2. Point `data/fruit_raw/` at the export root (train/valid/test or equivalent).
3. Run:
   ```powershell
   python src/split_fruit_groups.py --write
   ```

See `src/split_fruit_groups.py` — splits are **by source image group** so Roboflow augmented copies never leak across train/val/test.

## Checkpoint paths expected by inference

After training (or copying weights into `checkpoints/`):

| Model | File |
|---|---|
| Router | `checkpoints/router_v1_mobilenet_v3_small.pt` |
| Leaf | `checkpoints/leaf_v1_efficientnet_b0.pt` |
| Fruit | `checkpoints/fruit_v1_yolov8n.pt` |

See [`../checkpoints/README.md`](../checkpoints/README.md).
