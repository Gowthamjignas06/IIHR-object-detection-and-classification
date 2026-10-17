"""Held-out test evaluation report for the fruit YOLO detector.

Writes a JSON + Markdown report and compares it with the frozen v1 report.

Usage:
    python scripts/eval_fruit_test_report.py --checkpoint checkpoints/fruit_v2_yolov8n.pt
    python scripts/eval_fruit_test_report.py            # re-evaluates v1 into a NEW file

The default output is results/fruit_detector_test_report_<checkpoint stem>.json,
so the committed v1 report (results/fruit_detector_test_report.json) is never
overwritten. Existing output files are refused unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from train_fruit_yolo import (  # noqa: E402
    repo_root,
    resolve_data_yaml,
    resolve_device,
    write_resolved_data_yaml,
)

V1_REPORT = Path("results") / "fruit_detector_test_report.json"
V1_CHECKPOINT = Path("checkpoints") / "fruit_v1_yolov8n.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Error analysis of the v1 checkpoint (scripts/analyze_fruit_confusion.py,
# scripts/analyze_swv_box_areas.py). Kept for reference; not recomputed here.
V1_KNOWN_LIMITATIONS = {
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


def _count_images(data_yaml: Path, split: str) -> int:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    split_dir = Path(payload["path"]) / payload[split]
    return sum(1 for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _run_split(
    model: YOLO, data_yaml: Path, split: str, config: dict, device: str | int
) -> dict:
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(config["training"]["imgsz"]),
        batch=int(config["training"]["batch"]),
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    names = model.names
    per_class = []
    for i in sorted(names):
        p, r, ap50, ap = box.class_result(i)
        per_class.append(
            {
                "class": names[i],
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "map50": round(float(ap50), 4),
                "map50_95": round(float(ap), 4),
            }
        )
    return {
        "split": split,
        "n_images": _count_images(data_yaml, split),
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


def _compare_to_v1(test_m: dict, v1_report_path: Path) -> dict | None:
    if not v1_report_path.exists():
        return None
    v1_test = json.loads(v1_report_path.read_text(encoding="utf-8"))["test"]
    v1_rows = {row["class"]: row for row in v1_test["per_class"]}
    overall = {
        key: {
            "v1": v1_test[key],
            "new": test_m[key],
            "delta": round(test_m[key] - v1_test[key], 4),
        }
        for key in ("map50", "map50_95", "precision_mean", "recall_mean")
    }
    per_class = []
    for row in test_m["per_class"]:
        v1_row = v1_rows.get(row["class"])
        if v1_row is None:
            continue
        per_class.append(
            {
                "class": row["class"],
                **{
                    f"delta_{key}": round(row[key] - v1_row[key], 4)
                    for key in ("precision", "recall", "map50", "map50_95")
                },
            }
        )
    return {"v1_report": v1_report_path.as_posix(), "test_overall": overall, "test_per_class": per_class}


def _checkpoint_info(weights: Path, root: Path, best_epoch: int | None) -> dict:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args") or {}
    try:
        rel = weights.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = weights.as_posix()
    return {
        "path": rel,
        "model_version": weights.stem,
        "trained_at": ckpt.get("date"),
        "ultralytics_version": ckpt.get("version"),
        "best_epoch": best_epoch,
        "epochs_cap": train_args.get("epochs"),
        "patience": train_args.get("patience"),
        "train_val_metrics": ckpt.get("train_metrics"),
    }


def _write_markdown(report: dict, path: Path) -> None:
    val_m = report["val"]
    test_m = report["test"]
    ckpt = report["checkpoint"]
    lines = [
        f"# Fruit YOLO Detector — Held-Out Test Evaluation Report ({ckpt['model_version']})",
        "",
        f"**Date:** {report['generated_at']}  ",
        f"**Checkpoint:** `{ckpt['path']}`  ",
        f"**Device:** {report['device']}",
        "",
        "## Checkpoint summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Trained at | {ckpt['trained_at']} |",
        f"| Best epoch | {ckpt['best_epoch'] if ckpt['best_epoch'] is not None else 'not recorded'} |",
        f"| Epoch cap / patience | {ckpt['epochs_cap']} / {ckpt['patience']} |",
        "",
        f"## Test-set overall metrics (held-out, n={test_m['n_images']} images)",
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
    comparison = report.get("comparison_to_v1")
    if comparison:
        lines.extend(["", f"## Comparison with v1 (`{comparison['v1_report']}`)", ""])
        lines.append("| Metric | v1 | This run | Δ |")
        lines.append("|---|---:|---:|---:|")
        for key, row in comparison["test_overall"].items():
            lines.append(f"| {key} | {row['v1']:.4f} | {row['new']:.4f} | {row['delta']:+.4f} |")
        lines.extend(["", "| Class | ΔP | ΔR | ΔmAP@0.5 | ΔmAP@0.5:0.95 |", "|---|---:|---:|---:|---:|"])
        for row in comparison["test_per_class"]:
            lines.append(
                f"| {row['class']} | {row['delta_precision']:+.4f} | {row['delta_recall']:+.4f} | "
                f"{row['delta_map50']:+.4f} | {row['delta_map50_95']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Related artifacts",
            "",
            "- v1 known limitations: see `known_limitations_v1_reference` in the JSON report",
            "- Confusion analysis: `scripts/analyze_fruit_confusion.py`",
            f"- Machine-readable report: `{path.with_suffix('.json').name}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fruit detector held-out test report.")
    parser.add_argument("--checkpoint", type=Path, default=None, help=f"Default: {V1_CHECKPOINT}")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (Markdown written alongside). "
        "Default: results/fruit_detector_test_report_<checkpoint stem>.json",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | 0")
    parser.add_argument("--best-epoch", type=int, default=None, help="Recorded in the report.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing report.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = repo_root()
    weights = args.checkpoint or V1_CHECKPOINT
    if not weights.is_absolute():
        weights = root / weights
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    out_json = args.out or Path("results") / f"fruit_detector_test_report_{weights.stem}.json"
    if not out_json.is_absolute():
        out_json = root / out_json
    out_json = out_json.with_suffix(".json")
    out_md = out_json.with_suffix(".md")
    v1_report = (root / V1_REPORT).resolve()
    if out_json.resolve() == v1_report:
        raise ValueError(f"Refusing to write over the frozen v1 report {V1_REPORT}.")
    for path in (out_json, out_md):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists. Choose another --out or pass --overwrite.")

    config = yaml.safe_load((root / "configs" / "fruit_yolo.yaml").read_text(encoding="utf-8"))
    data_yaml = write_resolved_data_yaml(resolve_data_yaml(config))
    device = resolve_device(args.device)

    model = YOLO(str(weights))
    val_m = _run_split(model, data_yaml, "val", config, device)
    test_m = _run_split(model, data_yaml, "test", config, device)
    flagged = _val_test_flags(val_m, test_m, threshold=0.05)

    report = {
        "report_version": "1.1",
        "generated_at": str(date.today()),
        "status": "generated_unreviewed",
        "device": str(device),
        "checkpoint": _checkpoint_info(weights, root, args.best_epoch),
        "known_limitations_v1_reference": V1_KNOWN_LIMITATIONS,
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
        "comparison_to_v1": _compare_to_v1(test_m, v1_report),
    }

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
    comparison = report["comparison_to_v1"]
    if comparison:
        print()
        print("=== vs v1 (test) ===")
        for key, row in comparison["test_overall"].items():
            print(f"{key:<16} v1={row['v1']:.4f} new={row['new']:.4f} delta={row['delta']:+.4f}")
    print()
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
