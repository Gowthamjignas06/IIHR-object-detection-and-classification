"""Verify leaf checkpoint: on-disk weights match best_macro_f1 metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from datasets import build_leaf_dataloader, load_class_map  # noqa: E402
from evaluate import evaluate_leaf, repo_root  # noqa: E402
from models.leaf_classifier import build_leaf_classifier  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def verify_checkpoint(checkpoint_path: Path) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")
    _log(f"Loading checkpoint (CPU map): {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ck["class_names"]
    root = repo_root()
    _log(f"Stored best_macro_f1={float(ck['best_macro_f1']):.6f}")

    _log("Building model and loading weights...")
    model = build_leaf_classifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device)
    _log("Building val dataloader...")
    loader = build_leaf_dataloader("val", batch_size=16, shuffle=False, root=root)
    _log(f"Running live val eval ({len(loader.dataset)} images, {len(loader)} batches)...")
    live_metrics = evaluate_leaf(model, loader, device, class_names)
    _log("Live val eval complete.")

    stored_f1 = float(ck["best_macro_f1"])
    live_f1 = float(live_metrics["macro_f1"])
    stored_phase = int(ck.get("best_phase", ck.get("phase", -1)))
    stored_epoch = int(ck.get("best_epoch", ck.get("epoch", -1)))
    delta = abs(live_f1 - stored_f1)

    _log("=== Leaf checkpoint verification ===")
    _log(f"Path: {checkpoint_path}")
    _log(f"Stored best: phase {stored_phase} epoch {stored_epoch} macro-F1={stored_f1:.6f}")
    _log(f"Live val eval: macro-F1={live_f1:.6f} (delta={delta:.6f})")
    _log(
        f"Early blight precision: stored={ck['val_metrics']['per_class']['Early blight']['precision']:.4f} "
        f"live={live_metrics['per_class']['Early blight']['precision']:.4f}"
    )

    history = ck.get("metrics_history", [])
    if history:
        best_row = max(history, key=lambda row: row["macro_f1"])
        _log(
            f"History peak: phase {best_row['phase']} epoch {best_row['epoch']} "
            f"macro-F1={best_row['macro_f1']:.6f}"
        )
        if abs(best_row["macro_f1"] - stored_f1) > 1e-4:
            _log("WARN: stored best_macro_f1 does not match history peak.")

    ok = delta < 1e-4
    if ok:
        _log("PASS: checkpoint weights match stored best_macro_f1.")
    else:
        _log("FAIL: checkpoint weights do NOT match stored best_macro_f1.")
    return 0 if ok else 1


if __name__ == "__main__":
    path = (
        repo_root() / "checkpoints" / "leaf_v1_efficientnet_b0.pt"
        if len(sys.argv) == 1
        else Path(sys.argv[1])
    )
    raise SystemExit(verify_checkpoint(path))
