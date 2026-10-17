# CLAUDE.md

Tomato disease diagnosis pipeline (capstone). Photo → router (leaf | fruit) → leaf classifier or fruit detector → unified JSON + disease guide. Phases 1–2 (software) done; phase 3 (sensors/actuators) is out of scope.

Branches: `main` (10 leaf + 8 fruit classes, this file). `4-class-classifier` is a variant whose `src/four_class_infer.py` maps the same v1 models to 4 classes. It has its own CLAUDE.md.

## Working rules (from the project owner — always follow)

1. **Never overwrite v1 artifacts.** `checkpoints/*_v1_*.pt`, everything in `results/` (including `inference_log.json` and its example images) and `report_version1.md` are the frozen v1 baseline. New checkpoints/reports get a `v2` suffix.
2. **Unified JSON schema stays backward compatible.** Never rename, remove or redefine the seven original keys. New keys may be added.
3. **Small steps.** Show a plan before large edits (unless the owner says to just do it), and use one commit per logical fix. Don't push unless asked.
4. **After any model change**, rerun that model's eval and compare against v1 (see Metrics).
5. **The Google Drive photos in `data/external_validation/` are for validation only.** Never train, fine-tune, augment with them, or pick thresholds/hyperparameters on them.

### Built-in guards (keep them)
- `src/train.py` and `src/train_fruit_yolo.py` refuse to start if `checkpoints/<config filename>` exists, unless `--overwrite` is passed. Every config still names a v1 file, so train from a copy with a v2 `checkpoint.filename`.
- `scripts/eval_fruit_test_report.py` writes `results/fruit_detector_test_report_<ckpt stem>.json/.md`, refuses to write over the v1 report or any existing file, and adds `comparison_to_v1`.
- Still unguarded: `split_fruit_groups.py --write` (runs `rmtree` on `data/splits/fruit/`), `split_leaf_manifest.py --write` and `build_router_manifest.py --write` overwrite the committed splits.

## Unified output schema

```json
{
  "domain": "leaf | fruit",
  "class": "string",
  "confidence": 0.0,
  "severity_pct": 0.0,
  "gradcam_overlay": "path | null",
  "detections": "[{class, confidence, bbox:[x1,y1,x2,y2] px}] | null",
  "model_version": "checkpoint filename stem, e.g. leaf_v1_efficientnet_b0",

  "router_confidence": 0.0,
  "warnings": ["no_fruit_detected"],
  "advice": {"condition", "category", "contagious", "explanation", "cause", "symptoms",
             "why_it_matters", "urgency", "urgency_reason",
             "treatment_plan": {"cultural", "organic", "chemical"},
             "action_steps": {"immediate", "this_week", "prevention"},
             "notes", "get_help", "disclaimer", "knowledge_base_version"}
}
```
The first seven keys are the v1 contract and `tests/test_schema.py` enforces them. When YOLO finds nothing, fruit results still return `class` = `"Healthy_Tomato"` with confidence 0.0 for compatibility, plus `warnings: ["no_fruit_detected"]`. The app displays this as "No fruit detected". `advice` is omitted with `--no-advice` / `with_advice=False`.

## Layout

The git repo is **this directory**, not the parent `IIHR/`.

```
src/
  infer_pipeline.py   TomatoInferencePipeline (infer(path), infer_image(PIL)) + CLI
  advice.py           builds the "advice" key from knowledge/disease_advice.yaml; markdown renderer
  app.py              Gradio UI on 127.0.0.1:7860 (Diagnose button, Disease guide section)
  train.py            router + leaf training (two-phase fine-tune, config-driven)
  train_fruit_yolo.py Ultralytics wrapper; resolve/write_resolved_data_yaml, resolve_device
  evaluate.py         metrics, checkpoint loaders, repo_root(); CLI --model router|leaf (prints only)
  datasets.py, transforms.py, gradcam.py, severity.py, models/
  split_leaf_manifest.py / build_router_manifest.py / split_fruit_groups.py
knowledge/disease_advice.yaml   one entry per model class + _no_detection (draft, reviewed: false)
scripts/              eval_fruit_test_report, eval_leaf_test_report, eval_external_validation,
                      verify_leaf_checkpoint, analyze_*
tests/                test_schema.py (JSON contract, severity), test_advice.py (KB coverage, urgency)
configs/              router.yaml, leaf.yaml, fruit_yolo.yaml (all name v1 checkpoint files)
checkpoints/          v1 weights (tracked in git)
data/splits/          leaf/ router/ CSV manifests; fruit/ full YOLO layout (images+labels committed)
data/external_validation/   Drive field photos (gitignored, validation only)
results/              v1 test reports (frozen), external_validation/ report, app outputs (gitignored)
```

## Data flow

1. `load_image`: PIL open → `ImageOps.exif_transpose` → RGB. Relative paths resolve against the CWD first, then the repo root.
2. Router: CLAHE → resize 224 → ImageNet norm → MobileNetV3-Small → argmax (`leaf`=0, `fruit`=1). Confidence goes into `router_confidence`. There is **no rejection threshold** (see validation findings).
3. Leaf: EfficientNet-B0 softmax (10 classes; names from the checkpoint) → Grad-CAM at `<gradcam_dir>/<stem>_gradcam.jpg` → `leaf_severity_pct` (HSV green mask + Lab deviation).
4. Fruit: YOLO gets the same in-memory image (`conf=0.25, iou=0.7`, device passed through); class names come from `model.names` → `fruit_severity_pct` = union area of disease boxes ÷ union of healthy+disease boxes (or bounding rect of all boxes, or the full image for a single disease box).
5. `warnings`, `router_confidence`, `advice` are added in `infer_image`.
6. Checkpoints default to the v1 files; override with `router_/leaf_/fruit_checkpoint=` or `--router-checkpoint`, etc. A missing checkpoint raises `FileNotFoundError` (no silent `runs/` fallback).

## Commands

Run from the repo root with the venv: `.venv/bin/python …` or `source .venv/bin/activate`.

```bash
python -m pytest tests/                                  # 48 tests, ~3 s on CPU
python -u src/infer_pipeline.py --json <img> [...] [--device cpu] [--fruit-checkpoint …] [--no-advice]
python src/app.py                                         # http://127.0.0.1:7860

python src/train.py --config configs/<v2 copy>.yaml       # refuses existing checkpoint files
python src/train_fruit_yolo.py --config configs/<v2 copy>.yaml
python scripts/eval_fruit_test_report.py --checkpoint checkpoints/fruit_v2_yolov8n.pt [--device auto]
python scripts/eval_leaf_test_report.py --checkpoint …    # prints only; needs PlantVillage
python scripts/eval_external_validation.py [--save-overlays]   # Drive photos, inference only
.venv/bin/gdown --folder <drive url> -O data/external_validation   # Google throttles repeat downloads
```

## Environment (this Mac)

- Python 3.12 (Homebrew) venv at `.venv/`, CPU-only torch on Apple M5 (`resolve_device("auto")` → CPU; MPS unused). `requirements-lock.txt` has the exact verified versions (torch 2.14.1, ultralytics 8.4.123, gradio 6.30.0, numpy 2.1.3). `requirements.txt` sets lower bounds only.
- **Leaf/router manifests contain absolute Windows paths** (`C:/Users/hp/...`) and `data/leaf_raw/` is empty. Leaf/router eval and training can't run until PlantVillage is placed in `data/leaf_raw/PlantVillage/` **and** the manifest paths are rewritten as relative paths. Keep the same rows, so the v1 test set is unchanged; don't re-split.
- `app.py` writes to `results/inference_results/app_log.json` + JPEGs (gitignored). The committed `inference_log.json` is a frozen v1 example.

## Conventions

- `from __future__ import annotations`, `pathlib`, type hints, argparse `main() -> int` + `raise SystemExit(main())`. Scripts add `src/` to `sys.path`; there's no package install.
- Configs: YAML with `model`, `experiment_name`, `data`, `training`, `checkpoint.filename`.
- Leaf checkpoint dict: `model_state_dict, class_names, config, val_metrics, best_phase, best_epoch, metrics_history, best_macro_f1, completed_phases`. Router: `model_state_dict, config, val_metrics, phase, epoch`. Fruit: Ultralytics `.pt` (best epoch is stripped; pass `--best-epoch` to the eval script).
- Leaf class names are human-readable and sorted (`data/splits/leaf/class_map.json`). Fruit class names are Roboflow's (`data/splits/fruit/data.yaml`). Knowledge-base keys must match both exactly (`tests/test_advice.py` checks).
- Splits: 70/15/15, seed 42; fruit is grouped by Roboflow source stem (before `.rf.`).

## Metrics

### v1 baseline (held-out test, from results/)
| Model | Test | Val |
|---|---|---|
| Router `router_v1_mobilenet_v3_small` | acc 1.000 (2,611 imgs; from report_version1.md, no report file) | acc 1.000 |
| Leaf `leaf_v1_efficientnet_b0` | macro-F1 **0.9942**, acc 0.9942 (n=2,401) | macro-F1 0.9930 |
| Fruit `fruit_v1_yolov8n` | mAP50 **0.8067**, mAP50-95 0.4714, P 0.8105, R 0.7402 (n=210) | mAP50 0.7611 |

**Environment drift:** v1 fruit re-evaluated on this Mac (CPU, ultralytics 8.4.123) gives test mAP50 **0.8053** / mAP50-95 0.4653 / P 0.8095 / R 0.7402 (val 0.7584). Compare v2 fruit runs against these re-evaluated numbers, so the ~0.002–0.006 environment difference isn't counted as a model change.

Leaf test F1 per class: Bacterial spot .9969 · Early blight .9899 · Healthy 1.0 · Late blight .9930 · Leaf Mold .9930 · Septoria .9962 · Spider mites .9862 · Target Spot .9905 · TYLCV .9958 · Mosaic 1.0.
Fruit test per class (P/R/mAP50): Anthracnose .779/.923/.851 · BER .862/.923/.918 · Catfaced .785/.850/.830 · Fruit_Cracking .698/.576/.689 · Healthy .824/.618/.760 · Late_Blight .875/.762/.865 · Mold .778/.561/.688 · SWV .882/.708/.853.

### External validation (Drive field photos, v1 models, inference only)
See `results/external_validation/report.md`. Findings:
- **Real tomato fruit photos** (`fruit pics`): all routed to fruit, most called Healthy_Tomato. The few disease calls are low confidence (0.28–0.62) and visually doubtful: smooth red fruit called Catfaced, an overripe shrivelled fruit called SWV, a dry leaf called Mold.
- **Non-tomato photos** (`Article related_other pics`: group photos, landscapes): almost all routed to **fruit with ~0.99 router confidence**, and several got "Mold" boxes. The router learned "PlantVillage lab photo vs anything else", so there's no out-of-scope rejection.
- Implications: the router needs field-photo training data plus an out-of-scope class/threshold, and disease calls need a minimum confidence. **Pick any such threshold on the internal val split, never on the Drive set.**

## Known open issues
- Leaf severity: the HSV green mask (H 25–95 on OpenCV's 0–180 scale) leaves brown/necrotic lesions out of both the numerator and the denominator, so it under-reports blights. Needs leaf images to validate the fix.
- Grad-CAM files are named by image stem (same stem → overwritten).
- PlantVillage leaf split is per image (possible near-duplicate leakage); its lab backgrounds mean 0.994 likely overstates field accuracy.
- `knowledge/disease_advice.yaml` is a draft (`reviewed: false`) until an agronomist (IIHR/KVK) reviews it.
- Data label noise: e.g. test image `Fruit_Cracking11_*` is annotated Catfaced.
