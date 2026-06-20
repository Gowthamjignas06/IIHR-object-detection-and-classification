"""Held-out test evaluation report for the leaf classifier checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from evaluate import (  # noqa: E402
    evaluate_leaf_checkpoint,
    print_leaf_metrics,
    repo_root,
)


def late_blight_fp_to_early_blight(metrics: dict) -> int:
    class_names = metrics["class_names"]
    confusion = metrics["confusion"]
    eb_idx = class_names.index("Early blight")
    lb_idx = class_names.index("Late blight")
    return int(confusion[lb_idx][eb_idx])


def main() -> int:
    checkpoint = repo_root() / "checkpoints" / "leaf_v1_efficientnet_b0.pt"
    results = evaluate_leaf_checkpoint(checkpoint, splits=("val", "test"))
    val_m = results["val"]
    test_m = results["test"]

    print("=== Leaf classifier — held-out TEST evaluation ===")
    print(f"Checkpoint: {checkpoint}")
    print()
    print_leaf_metrics(test_m, "test")
    print()

    eb_val = val_m["per_class"]["Early blight"]
    eb_test = test_m["per_class"]["Early blight"]
    lb_eb_val = late_blight_fp_to_early_blight(val_m)
    lb_eb_test = late_blight_fp_to_early_blight(test_m)

    print("=== Early blight & Late blight -> Early blight FP ===")
    print(f"{'Metric':<32} {'Baseline':>10} {'Val (best)':>12} {'Test':>10}")
    print("-" * 68)
    print(
        f"{'Early blight precision':<32} {0.7444:>10.4f} "
        f"{eb_val['precision']:>12.4f} {eb_test['precision']:>10.4f}"
    )
    print(
        f"{'Late blight -> Early blight FP':<32} {19:>10} "
        f"{lb_eb_val:>12} {lb_eb_test:>10}"
    )
    print()

    warn_threshold = 0.03
    severe_threshold = 0.05
    flagged: list[str] = []
    print("=== Per-class val vs test (F1 / precision / recall deltas) ===")
    print(f"{'Class':<40} {'Val F1':>7} {'Test F1':>7} {'dF1':>7}  Flag")
    print("-" * 72)
    for name in val_m["class_names"]:
        v = val_m["per_class"][name]
        t = test_m["per_class"][name]
        df1 = t["f1"] - v["f1"]
        dprec = t["precision"] - v["precision"]
        drec = t["recall"] - v["recall"]
        flag = ""
        drops = []
        if df1 < -warn_threshold:
            drops.append(f"F1 {df1:+.4f}")
        if dprec < -warn_threshold:
            drops.append(f"Prec {dprec:+.4f}")
        if drec < -warn_threshold:
            drops.append(f"Rec {drec:+.4f}")
        if drops:
            severity = "SEVERE" if any(
                df1 < -severe_threshold
                or dprec < -severe_threshold
                or drec < -severe_threshold
                for _ in [0]
            ) else "WARN"
            flag = f"{severity}: " + ", ".join(drops)
            flagged.append(f"{name}: {flag}")
        print(f"{name:<40} {v['f1']:>7.4f} {t['f1']:>7.4f} {df1:>+7.4f}  {flag}")

    print()
    if flagged:
        print("FLAG: test performance dropped >0.03 below val for:")
        for line in flagged:
            print(f"  - {line}")
    else:
        print("No class dropped >0.03 on F1/precision/recall from val to test.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
