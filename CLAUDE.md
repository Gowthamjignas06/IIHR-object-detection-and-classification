# CLAUDE.md — `4-class-classifier` branch

Four-class tomato diagnostic: Early blight · Blossom_End_Rot · Healthy leaf · Healthy plant. It is built **without retraining** on top of the same three v1 checkpoints as `main` (router → leaf EfficientNet-B0 / fruit YOLOv8n). `main` has the full 10 leaf + 8 fruit pipeline and its own CLAUDE.md; most shared code (infer_pipeline, severity, advice, train, eval) is kept in sync by cherry-picking from `main`.

## Working rules (from the project owner — always follow)

1. **Never overwrite v1 artifacts:** `checkpoints/*_v1_*.pt`, everything in `results/` (including `four_class_test_report.*`, `inference_log.json` and the example images), and `report_version1.md`. New ones get a `v2` suffix.
2. **Output schemas stay backward compatible; only add keys.** On this branch the contract is `FourClassDiagnosticClassifier.classify()`'s output (see below), and `src/infer_pipeline.py` keeps main's unified schema.
3. Small steps, a plan before large edits, one commit per logical fix. No pushing unless asked.
4. After any model change, rerun the evals and compare with v1.
5. **The Google Drive photos (`data/external_validation/`) are for validation only.** Never train or tune on them.
6. Make shared-file fixes on `main` first, then cherry-pick them here. Hand-port only `src/app.py`, `src/four_class_infer.py`, README and this file.

## Four-class output schema

```json
{
  "predicted_class": "Early blight | Blossom_End_Rot | Healthy leaf | Healthy plant",
  "confidence": 0.0, "domain": "leaf | fruit", "router_confidence": 0.0,
  "class_scores": {"Early blight": 0.0, "Blossom_End_Rot": 0.0, "Healthy leaf": 0.0, "Healthy plant": 0.0},
  "detections": "[{class, confidence, bbox}] | null", "raw_leaf_top3": "[{class, score}] | null",
  "model_versions": {"router": "...", "leaf": "...", "fruit": "..."},

  "severity_pct": 0.0,
  "warnings": ["no_fruit_detected | other_condition_detected | other_leaf_condition_detected"],
  "other_conditions": [{"class": "Catfaced", "confidence": 0.86}],
  "advice": {"...": "same structure as main"}
}
```
The first eight keys are the contract, enforced by `tests/test_four_class.py`. Known fallback behaviour, kept for compatibility:
- fruit with no target box → `Healthy plant` 1.0, even when YOLO saw nothing (`no_fruit_detected`) or only other defects (`other_condition_detected`);
- leaf → whichever of Early blight / Healthy scores higher, even when the 10-class top-1 is another disease (`other_leaf_condition_detected`).

The app headline uses the warnings ("No fruit detected", "Other condition: X"). Advice covers `other_conditions[0]` when it's set.

## Layout (branch-specific parts)
- `src/four_class_infer.py`: `FourClassDiagnosticClassifier` (`classify(path)`, `classify_image(PIL)`) + CLI.
- `src/app.py`: four-class Gradio UI (Classify button, score panel, Disease guide).
- `configs/four_class.yaml`: descriptive config (target classes, thresholds); the code doesn't read it.
- `results/four_class_test_report.{json,md}`: frozen v1 four-class report.
- Shared with main: `src/infer_pipeline.py`, `severity.py`, `advice.py`, `knowledge/disease_advice.yaml`, `train*.py`, `scripts/`, `tests/test_schema.py`, `tests/test_advice.py`.

## Commands
```bash
.venv/bin/python -m pytest tests/                       # 53 tests
.venv/bin/python src/four_class_infer.py <img> [...] [--json] [--device cpu] [--fruit-conf 0.20] [--no-advice]
.venv/bin/python src/app.py                             # http://127.0.0.1:7860
.venv/bin/python src/infer_pipeline.py --json <img>      # full 18-class pipeline (main's schema)
```
Fruit YOLO threshold here is `conf=0.20` (main's pipeline uses 0.25).

## Environment
The same as main: Python 3.12 venv, CPU-only on Apple M5, exact versions in `requirements-lock.txt`. Leaf/router manifests hold absolute Windows paths and PlantVillage isn't present, so leaf/router eval can't run until the data is added and the paths are made relative.

## v1 metrics (from results/four_class_test_report.json)
- Router: acc 1.000 (val/test).
- Leaf (10-class model, test n=2,401: macro-F1 0.9942). Early blight P 1.000 / R 0.980 / F1 0.9899 (n=150). Healthy leaf P/R/F1 1.000 (n=238).
- Fruit (YOLO test, mAP50 0.8067 overall). Blossom_End_Rot P .862 / R .923 / mAP50 .918. Healthy plant (Healthy_Tomato) P .824 / R .618 / mAP50 .760.
- v1 fruit re-evaluated on this Mac's CPU (ultralytics 8.4.123): mAP50 0.8053. Compare v2 against that.

## Known open issues
- On the fruit test images, many diseased fruits come out as `predicted_class: Healthy plant` 1.0 (e.g. anthracnose13, catfaced_10, spot21). The warnings expose this, but a real fix would add a 5th "Other condition" class to the output, which changes the contract and needs the owner's decision.
- Out-of-scope photos (people, landscapes) are routed with ~0.99 confidence (see main's `results/external_validation/report.md`).
- Leaf severity under-reports brown lesions; the knowledge base is a draft until an agronomist reviews it.
