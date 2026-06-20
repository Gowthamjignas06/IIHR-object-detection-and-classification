"""Analyze Early blight confusion from a leaf checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from evaluate import repo_root  # noqa: E402


def analyze_early_blight(checkpoint_path: Path) -> None:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = ck["val_metrics"]
    class_names: list[str] = metrics["class_names"]
    confusion = metrics["confusion"]
    eb_idx = class_names.index("Early blight")

    print("=== Early blight confusion analysis ===")
    print(f"Checkpoint: phase {ck.get('phase')} epoch {ck.get('epoch')}")
    eb = metrics["per_class"]["Early blight"]
    print(
        f"Early blight: precision={eb['precision']:.4f} "
        f"recall={eb['recall']:.4f} f1={eb['f1']:.4f} support={eb['support']}"
    )
    print()

    row = confusion[eb_idx]
    row_total = sum(row)
    print("True Early blight misclassified AS (row):")
    for idx, count in sorted(
        ((i, row[i]) for i in range(len(class_names)) if i != eb_idx and row[i] > 0),
        key=lambda x: -x[1],
    ):
        print(f"  -> {class_names[idx]}: {count} ({count / row_total:.1%} of true EB)")
    print(f"  -> correct Early blight: {row[eb_idx]} ({row[eb_idx] / row_total:.1%})")
    print()

    col_counts = [
        (i, confusion[true_idx][eb_idx])
        for true_idx in range(len(class_names))
        for i in [true_idx]
        if confusion[true_idx][eb_idx] > 0 and true_idx != eb_idx
    ]
    col_total = sum(confusion[true_idx][eb_idx] for true_idx in range(len(class_names)))
    fp_total = col_total - row[eb_idx]
    print("Other classes misclassified AS Early blight (column false positives):")
    for idx, count in sorted(col_counts, key=lambda x: -x[1]):
        print(f"  {class_names[idx]} -> Early blight: {count}")
    print(f"  Total predicted Early blight: {col_total} (TP={row[eb_idx]}, FP={fp_total})")


if __name__ == "__main__":
    path = (
        repo_root() / "checkpoints" / "leaf_v1_efficientnet_b0.pt"
        if len(sys.argv) == 1
        else Path(sys.argv[1])
    )
    analyze_early_blight(path)
