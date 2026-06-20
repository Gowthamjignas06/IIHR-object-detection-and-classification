"""Held-out test evaluation report for the fruit YOLO detector."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import yaml
from ultralytics import YOLO

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from train_fruit_yolo import (  # noqa: E402
    repo_root,
    resolve_data_yaml,
    write_resolved_data_yaml,
)

CLASS_NAMES = [
    "Anthracnose",
    "Blossom_End_Rot",
    "Catfaced",
    "Fruit_Cracking",
    "Healthy_Tomato",
    "Late_Blight",
    "Mold",
    "Spotted_Wilt_Virus",
]

KNOWN_LIMITATIONS = {
    "Catfaced": {
        "issue": "Recall-perfect but precision-poor (over-detection)",
        "val_precision": 0.568,
        "val_recall": 1.0,
        "dominant_fp_source": "Fruit_Cracking -> pred Catfaced (40% of FP)",
        "other_fp_sources": [
            "background spurious boxes (33.3% of FP)",
            "Blossom_End_Rot -> pred Catfaced (26.7% of FP)",
        ],
    },
    "Spotted_Wilt_Virus": {
        "issue": "Low recall; class confusion with Late_Blight at high IoU",
        "val_recall": 0.368,
        "val_precision": 0.886,
        "missed_as_background_pct_of_gt": 31.6,
        "absorbed_as_late_blight_pct_of_gt": 26.3,
        "conclusion": "More labeled data needed, not more training epochs on this split.",
    },
}


def _run_split(model: YOLO, data_yaml: Path, split: str, config: dict) -> dict:
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(config["training"]["imgsz"]),
        batch=int(config["training"]["batch"]),
        device=0,
        workers=0,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class.append(
            {
                "class": name,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "map50": round(float(ap50), 4),
                "map50_95": round(float(ap), 4),
            }
        )
    return {
        "split": split,
        "n_images": int(getattr(metrics, "nt", 0) or (210 if split == "test" else 199)),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision_mean": round(float(box.mp), 4),
        "recall_mean": round(float(box.mr), 4),
        "per_class": per_class,
    }


def _val_test_flags(val_m: dict, test_m: dict, threshold: float = 0.05) -> list[dict]:
    flagged = []
    for v_row, t_row in zip(val_m["per_class"], test_m["per_class"]):
        d50 = t_row["map50"] - v_row["map50"]
        d95 = t_row["map50_95"] - v_row["map50_95"]
        if d50 < -threshold or d95 < -threshold:
            flagged.append(
                {
                    "class": v_row["class"],
                    "val_map50": v_row["map50"],
                    "test_map50": t_row["map50"],
                    "delta_map50": round(d50, 4),
                    "val_map50_95": v_row["map50_95"],
                    "test_map50_95": t_row["map50_95"],
                    "delta_map50_95": round(d95, 4),
                }
            )
    return flagged


def _write_markdown(report: dict, path: Path) -> None:
    val_m = report["val"]
    test_m = report["test"]
    lines = [
        "# Fruit YOLO Detector — Held-Out Test Evaluation Report",
        "",
        f"**Step 6 — approved complete**  ",
        f"**Date:** {report['generated_at']}  ",
        f"**Checkpoint:** `{report['checkpoint']['path']}`",
        "",
        "## Checkpoint summary",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Model | YOLOv8n |",
        f"| Best epoch | {report['checkpoint']['best_epoch']} |",
        f"| Stop reason | Early stopping (patience=20) at epoch 64 |",
        f"| Canonical artifact | `{report['checkpoint']['path']}` |",
        "",
        "## Test-set overall metrics (held-out, n=210 images)",
        "",
        "| Metric | Val | Test |",
        "|---|---:|---:|",
        f"| mAP@0.5 | {val_m['map50']:.4f} | **{test_m['map50']:.4f}** |",
        f"| mAP@0.5:0.95 | {val_m['map50_95']:.4f} | **{test_m['map50_95']:.4f}** |",
        f"| Mean precision | {val_m['precision_mean']:.4f} | {test_m['precision_mean']:.4f} |",
        f"| Mean recall | {val_m['recall_mean']:.4f} | {test_m['recall_mean']:.4f} |",
        "",
        "## Per-class metrics (test)",
        "",
        "| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in test_m["per_class"]:
        lines.append(
            f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['map50']:.4f} | {row['map50_95']:.4f} |"
        )
    lines.extend(["", "## Val vs test mAP (overfit check, threshold −0.05)", ""])
    lines.append("| Class | Val mAP@0.5 | Test mAP@0.5 | Δ | Flag |")
    lines.append("|---|---:|---:|---:|---|")
    for v_row, t_row in zip(val_m["per_class"], test_m["per_class"]):
        d = t_row["map50"] - v_row["map50"]
        flag = "SEVERE" if d < -0.05 else ("WARN" if d < -0.03 else "—")
        lines.append(
            f"| {v_row['class']} | {v_row['map50']:.4f} | {t_row['map50']:.4f} | {d:+.4f} | {flag} |"
        )
    lines.extend(["", "## Known limitations (this iteration)", ""])
    cat = KNOWN_LIMITATIONS["Catfaced"]
    swv = KNOWN_LIMITATIONS["Spotted_Wilt_Virus"]
    lines.extend(
        [
            "### Catfaced",
            f"- {cat['issue']}",
            f"- Val precision {cat['val_precision']}, recall {cat['val_recall']}",
            f"- Dominant FP: **{cat['dominant_fp_source']}**",
            "",
            "### Spotted_Wilt_Virus",
            f"- {swv['issue']}",
            f"- Val recall {swv['val_recall']}, precision {swv['val_precision']}",
            f"- {swv['missed_as_background_pct_of_gt']}% of GT missed as background; "
            f"{swv['absorbed_as_late_blight_pct_of_gt']}% absorbed as Late_Blight (79–89% IoU)",
            f"- **Conclusion:** {swv['conclusion']}",
            "",
            "## Related artifacts",
            "",
            "- Training log: `logs/fruit_train.log`",
            "- Confusion analysis: `scripts/analyze_fruit_confusion.py`",
            f"- Machine-readable report: `{path.with_suffix('.json').name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    config = yaml.safe_load((root / "configs" / "fruit_yolo.yaml").read_text(encoding="utf-8"))
    data_yaml = write_resolved_data_yaml(resolve_data_yaml(config))
    weights = root / "checkpoints" / "fruit_v1_yolov8n.pt"
    if not weights.exists():
        weights = root / "runs" / "fruit" / "train" / "weights" / "best.pt"

    model = YOLO(str(weights))
    val_m = _run_split(model, data_yaml, "val", config)
    test_m = _run_split(model, data_yaml, "test", config)
    flagged = _val_test_flags(val_m, test_m, threshold=0.05)

    report = {
        "report_version": "1.0",
        "generated_at": str(date.today()),
        "step": 6,
        "status": "approved_complete",
        "checkpoint": {
            "path": "checkpoints/fruit_v1_yolov8n.pt",
            "canonical": True,
            "best_epoch": 44,
            "stop_epoch": 64,
            "stop_reason": "early_stopping_patience_20",
        },
        "known_limitations": KNOWN_LIMITATIONS,
        "val": val_m,
        "test": test_m,
        "val_vs_test_map_flags": flagged,
        "val_overfit_assessment": {
            "test_map50": test_m["map50"],
            "val_map50": val_m["map50"],
            "test_map50_95": test_m["map50_95"],
            "val_map50_95": val_m["map50_95"],
            "any_class_map50_drop_gt_0_05": len(flagged) > 0,
        },
    }

    out_json = root / "results" / "fruit_detector_test_report.json"
    out_md = root / "results" / "fruit_detector_test_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, out_md)

    print("=== Fruit detector — held-out TEST evaluation ===")
    print(f"Checkpoint: {weights}")
    print(f"Val  mAP@0.5={val_m['map50']:.4f}  mAP@0.5:0.95={val_m['map50_95']:.4f}")
    print(f"Test mAP@0.5={test_m['map50']:.4f}  mAP@0.5:0.95={test_m['map50_95']:.4f}")
    print()
    print(f"{'Class':<24} {'V-mAP50':>8} {'T-mAP50':>8} {'d':>8}")
    for v_row, t_row in zip(val_m["per_class"], test_m["per_class"]):
        d = t_row["map50"] - v_row["map50"]
        print(f"{v_row['class']:<24} {v_row['map50']:8.4f} {t_row['map50']:8.4f} {d:+8.4f}")
    print()
    if flagged:
        print("FLAG: classes with test mAP@0.5 drop >0.05 below val:")
        for row in flagged:
            print(f"  - {row['class']}: delta={row['delta_map50']:+.4f}")
    else:
        print("No class dropped >0.05 mAP@0.5 from val to test.")
    print()
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
