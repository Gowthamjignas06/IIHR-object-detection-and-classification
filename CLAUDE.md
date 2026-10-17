# CLAUDE.md

Tomato disease diagnosis pipeline (capstone). Photo → router (leaf | fruit) → leaf classifier or fruit detector → unified JSON. Phases 1–2 (software) done; phase 3 (sensors/actuators) is out of scope.

## Working rules (from the project owner — always follow)

1. **Never overwrite v1 artifacts.** `checkpoints/*_v1_*.pt` and every file in `results/` are the frozen v1 baseline. New checkpoints/reports get a `v2` suffix (e.g. `checkpoints/leaf_v2_efficientnet_b0.pt`, `results/leaf_classifier_test_report_v2.json`).
2. **Unified JSON schema stays backward compatible.** The seven keys below must keep their names, types and meaning. New keys may be added; nothing gets removed or renamed.
3. **Small steps.** Show a plan before any large edit and wait for approval.
4. **After any model change**, rerun that model's eval and compare against the v1 numbers below / in `results/`.

### Traps that break rule 1 if you run code unmodified
- `src/train.py` saves to `checkpoints/<config.checkpoint.filename>`, and every config still names a **v1** file. It also overwrites mid-training each time val improves. Before training, make a v2 config (e.g. `configs/leaf_v2.yaml`) with a v2 `filename` and `experiment_name`.
- `src/train_fruit_yolo.py` copies `best.pt` over `checkpoints/fruit_v1_yolov8n.pt` (same reason). Use a v2 config.
- `scripts/eval_fruit_test_report.py` hardcodes the v1 checkpoint and **writes over** `results/fruit_detector_test_report.{json,md}`. Don't run it as-is.
- `split_fruit_groups.py --write` runs `rmtree` on `data/splits/fruit/` before rewriting it. `split_leaf_manifest.py --write` and `build_router_manifest.py --write` overwrite the committed manifests.

## Unified output schema (contract)

```json
{
  "domain": "leaf | fruit",
  "class": "string",
  "confidence": 0.0,
  "severity_pct": 0.0,
  "gradcam_overlay": "path | null",
  "detections": "[{class, confidence, bbox:[x1,y1,x2,y2] px}] | null",
  "model_version": "string, e.g. leaf_v1_efficientnet_b0"
}
```
Leaf: `gradcam_overlay` = absolute path, `detections` = null. Fruit: `gradcam_overlay` = null, `class` = highest-confidence box, or `"Healthy_Tomato"` with confidence 0.0 if nothing is detected. The schema is also documented in ARCHITECTURE.md §4 and report_version1.md §9.

## Layout

The git repo is **this directory** (`IIHR-object-detection-and-classification/`), not the parent `IIHR/`.

```
src/
  infer_pipeline.py   TomatoInferencePipeline + CLI (main entry point)
  app.py              Gradio UI on 127.0.0.1:7860; wraps the pipeline, saves panels + log
  train.py            router + leaf training (two-phase fine-tune, config-driven)
  train_fruit_yolo.py Ultralytics wrapper; also provides resolve/write_resolved_data_yaml
  evaluate.py         metrics, checkpoint loaders, repo_root(); CLI --model router|leaf (prints only)
  datasets.py         LeafDataset / RouterDataset (CSV manifests), class weights, samplers
  transforms.py       CLAHE + leaf transforms (also used for the router)
  gradcam.py          Grad-CAM on model.features[-1]
  severity.py         leaf HSV/Lab severity; fruit box-geometry severity
  models/             build_router (MobileNetV3-S), build_leaf_classifier (EffNet-B0)
  split_leaf_manifest.py / build_router_manifest.py / split_fruit_groups.py   data splits
scripts/              eval_leaf_test_report, eval_fruit_test_report, verify_leaf_checkpoint, analyze_*
configs/              router.yaml, leaf.yaml, fruit_yolo.yaml
checkpoints/          v1 weights (tracked in git despite checkpoints/README.md saying otherwise)
data/splits/          leaf/ router/ CSV manifests; fruit/ full YOLO layout (images+labels committed)
data/leaf_raw, data/fruit_raw   raw datasets (gitignored, not present on this machine)
results/              v1 test reports (JSON+MD), project_summary.json, Gradio outputs
```
Docs: README.md (quick start), ARCHITECTURE.md (design spec), report_version1.md (long developer report), data/AUDIT.md (dataset audit), data/README.md (data setup).

## Data flow

1. `Image.open(...).convert("RGB")`. No EXIF transpose.
2. Router: CLAHE → resize 224 → ImageNet norm → MobileNetV3-Small → argmax over (`leaf`=0, `fruit`=1). The router's confidence is computed but thrown away, and there is no rejection threshold.
3. Leaf: same transform → EfficientNet-B0 softmax (10 classes, names come from the checkpoint's `class_names`) → Grad-CAM (second forward+backward) written to `results/gradcam_overlays/<stem>_gradcam.jpg` → `leaf_severity_pct` on the raw image.
4. Fruit: `YOLO.predict(path, conf=0.25, iou=0.7)` at 640 → detections → `fruit_severity_pct` (sum of disease box areas ÷ union of healthy boxes, else union of all boxes, else the full image).
5. Version strings are hardcoded constants in `infer_pipeline.py` (`*_MODEL_VERSION`), and checkpoint paths are built from them.

## Commands

Run everything from the repo root. Scripts add `src/` to `sys.path` themselves; there is no package install.

```bash
# Inference
python -u src/infer_pipeline.py --json <img> [<img> ...] [--device auto|cuda|cpu]
python src/app.py

# Data splits (need raw data; they OVERWRITE the committed splits)
python src/split_leaf_manifest.py --dry-run|--write
python src/split_fruit_groups.py  --dry-run|--write
python src/build_router_manifest.py [--write]

# Training (point at a v2 config first — see Traps)
python src/train.py --config configs/router.yaml [--smoke-test]
python src/train.py --config configs/leaf.yaml [--leaf-max-phase 1|2] [--resume ckpt] [--no-early-stopping]
python src/train_fruit_yolo.py --config configs/fruit_yolo.yaml [--smoke-test] [--epochs N]

# Eval
python src/evaluate.py --model router|leaf [--checkpoint path] [--splits val test]
python scripts/eval_leaf_test_report.py      # prints only; v1 checkpoint hardcoded
python scripts/verify_leaf_checkpoint.py [ckpt]
python scripts/eval_fruit_test_report.py     # OVERWRITES v1 report, device=0 hardcoded
```

## Environment status (this machine, macOS)

- No project Python env yet. System `python3` is 3.9 with no torch; the project targets 3.10+ (`requirements.txt`). Create a venv before running anything.
- The original setup was Windows + RTX 3050 Ti (4 GB). Here: no CUDA. `resolve_device("auto")` picks CPU (no MPS support). The fruit scripts hardcode `device=0` and will fail.
- **Every leaf row in `data/splits/leaf/*.csv` and `data/splits/router/*.csv` is an absolute Windows path** (`C:/Users/hp/Downloads/...`). Leaf/router training and eval can't run here until PlantVillage is downloaded and the manifests are regenerated or rewritten as relative paths. Fruit rows are relative and work.
- The fruit train/val/test images are in the repo, so fruit eval and inference work once torch and ultralytics are installed.

## Conventions

- Python with `from __future__ import annotations`, `pathlib.Path`, type hints, small functions, argparse `main() -> int` + `raise SystemExit(main())`.
- Paths resolve via `repo_root()` (defined in several modules). The CLI resolves relative image paths against the **repo root**, not the CWD.
- Configs are YAML with keys `model`, `experiment_name`, `data`, `training`, `checkpoint.filename`.
- Leaf checkpoints are dicts: `model_state_dict, class_names, config, val_metrics, best_phase, best_epoch, metrics_history, best_macro_f1, completed_phases`. Router: `model_state_dict, config, val_metrics, phase, epoch`. Fruit: a standard Ultralytics `.pt`.
- Class names: leaf uses human-readable names in sorted order (`data/splits/leaf/class_map.json`); fruit uses Roboflow names (`data/splits/fruit/data.yaml`).
- Splits: 70/15/15, seed 42. Fruit splits are grouped by the Roboflow source stem (the part before `.rf.`) to stop augmented copies leaking across splits.

## v1 baseline metrics (held-out test; compare all v2 work against these)

| Model | Checkpoint | Test | Val |
|---|---|---|---|
| Router | router_v1_mobilenet_v3_small.pt | acc 1.000 (2,611 imgs) | acc 1.000 |
| Leaf | leaf_v1_efficientnet_b0.pt | macro-F1 **0.9942**, acc 0.9942 (n=2,401) | macro-F1 0.9930 (P2 ep12) |
| Fruit | fruit_v1_yolov8n.pt | mAP50 **0.8067**, mAP50-95 0.4714, P 0.8105, R 0.7402 (n=210) | mAP50 0.7611, mAP50-95 0.4455 |

Router numbers come from report_version1.md only; no router report file exists in `results/`.

Leaf test F1 per class: Bacterial spot .9969 · Early blight .9899 · Healthy 1.0 · Late blight .9930 · Leaf Mold .9930 · Septoria .9962 · Spider mites .9862 · Target Spot .9905 · TYLCV .9958 · Mosaic 1.0.

Fruit test per class (P / R / mAP50):
Anthracnose .779/.923/.851 · Blossom_End_Rot .862/.923/.918 · Catfaced .785/.850/.830 · Fruit_Cracking .698/.576/.689 · Healthy_Tomato .824/.618/.760 · Late_Blight .875/.762/.865 · Mold .778/.561/.688 · Spotted_Wilt_Virus .882/.708/.853.
Known weak spots: Catfaced over-detection (val P .568, mostly from Fruit_Cracking), SWV low val recall (.368, often absorbed as Late_Blight), Mold and Fruit_Cracking recall.

Full numbers: `results/leaf_classifier_test_report.json`, `results/fruit_detector_test_report.json`.
