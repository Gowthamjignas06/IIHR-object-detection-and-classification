"""Analyze YOLO val confusion matrix for Catfaced FP and Spotted_Wilt_Virus FN patterns."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from train_fruit_yolo import resolve_data_yaml, write_resolved_data_yaml  # noqa: E402


def main() -> int:
    config_path = _REPO / "configs" / "fruit_yolo.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_yaml = write_resolved_data_yaml(resolve_data_yaml(config))
    weights = _REPO / "checkpoints" / "fruit_v1_yolov8n.pt"
    if not weights.exists():
        weights = _REPO / "runs" / "fruit" / "train" / "weights" / "best.pt"

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=int(config["training"]["imgsz"]),
        batch=int(config["training"]["batch"]),
        device=0,
        workers=0,
        plots=True,
        project=str(_REPO / "runs" / "fruit"),
        name="confusion_analysis",
        exist_ok=True,
        verbose=False,
    )

    cm_obj = metrics.confusion_matrix
    data_cfg = yaml.safe_load(
        (_REPO / "data" / "splits" / "fruit" / "data.yaml").read_text(encoding="utf-8")
    )
    names: list[str] = list(data_cfg["names"])
    matrix = cm_obj.matrix.copy()
    nc = len(names)
    bg = nc  # background row/col index

    print("=== YOLO val confusion matrix (detection, matched boxes) ===")
    print(f"Weights: {weights}")
    print(f"Orientation: rows=predicted, cols=true (Ultralytics detect CM)")
    print(f"Classes: {names}")
    print()

    def idx(name: str) -> int:
        return names.index(name)

    def print_true_class_breakdown(true_name: str) -> None:
        """Where GT instances of true_name went (FN / mis-pred)."""
        ti = idx(true_name)
        col = matrix[:, ti]
        total_gt = int(col.sum())
        tp = int(matrix[ti, ti])
        print(f"--- True {true_name} (GT instances={total_gt}) ---")
        print(f"  Correct TP (pred={true_name}): {tp}")
        errors = []
        for pi, pname in enumerate(names):
            if pi == ti:
                continue
            count = int(matrix[pi, ti])
            if count:
                errors.append((pname, count))
        fn_bg = int(matrix[bg, ti])
        if fn_bg:
            errors.append(("background / missed", fn_bg))
        errors.sort(key=lambda x: x[1], reverse=True)
        print("  GT misclassified as (predicted class):")
        for pname, count in errors:
            pct = 100 * count / total_gt if total_gt else 0
            print(f"    true {true_name} -> pred {pname}: {count} ({pct:.1f}% of GT)")
        print()

    def print_pred_class_fp_sources(pred_name: str) -> None:
        """Where false-positive predictions of pred_name come from."""
        pi = idx(pred_name)
        row = matrix[pi, :]
        total_pred = int(row.sum())
        tp = int(matrix[pi, pi])
        fp_total = total_pred - tp
        print(f"--- Predicted {pred_name} (total preds={total_pred}) ---")
        print(f"  TP: {tp}")
        print(f"  FP: {fp_total}")
        sources = []
        for ti, tname in enumerate(names):
            if ti == pi:
                continue
            count = int(matrix[pi, ti])
            if count:
                sources.append((tname, count))
        fp_bg = int(matrix[pi, bg])
        if fp_bg:
            sources.append(("background (no matching GT)", fp_bg))
        sources.sort(key=lambda x: x[1], reverse=True)
        print("  FP breakdown (true class of each wrong prediction):")
        for tname, count in sources:
            pct = 100 * count / fp_total if fp_total else 0
            print(f"    {tname} -> pred {pred_name}: {count} ({pct:.1f}% of FP)")
        print()

    print_pred_class_fp_sources("Catfaced")
    print_true_class_breakdown("Catfaced")
    print_true_class_breakdown("Spotted_Wilt_Virus")
    print_pred_class_fp_sources("Spotted_Wilt_Virus")

    print("=== Full count matrix (rows=predicted, cols=true) ===")
    header = names + ["background"]
    axis_label = "pred \\ true"
    print(f"{axis_label:24} " + " ".join(f"{h[:12]:>12}" for h in header))
    for pi, plabel in enumerate(names + ["background"]):
        row = matrix[pi]
        print(f"{plabel[:24]:24} " + " ".join(f"{int(v):12d}" for v in row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
