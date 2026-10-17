"""Shared training loop for router and leaf classifier."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from datasets import build_leaf_dataloader, build_router_dataloader, compute_class_weights, load_class_map
from evaluate import (
    check_phase1_minority_recall,
    evaluate_leaf,
    evaluate_router,
    macro_f1_still_rising,
    print_leaf_metrics,
    print_macro_f1_trajectory,
)
from models.leaf_classifier import build_leaf_classifier, set_leaf_train_phase
from models.router import build_router, set_router_train_phase

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def guard_checkpoint_path(path: Path, overwrite: bool) -> None:
    """Refuse to clobber an existing checkpoint (v1 checkpoints are frozen)."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Set checkpoint.filename in the config to a new "
            "v2 name (e.g. router_v2_mobilenet_v3_small.pt) or pass --overwrite."
        )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_router(config: dict, smoke_test: bool = False, overwrite: bool = False) -> Path:
    guard_checkpoint_path(CHECKPOINTS_DIR / config["checkpoint"]["filename"], overwrite)
    set_seed(config["training"]["seed"])
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Training device: {device}")

    root = repo_root()
    batch_size = config["data"]["batch_size"]
    image_size = config["data"]["image_size"]
    equal_sampling = config["data"].get("equal_domain_sampling", True)

    train_loader = build_router_dataloader(
        "train",
        batch_size=batch_size,
        image_size=image_size,
        equal_domain_sampling=equal_sampling,
        num_workers=config["data"].get("num_workers", 0),
        root=root,
    )
    val_loader = build_router_dataloader(
        "val",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        equal_domain_sampling=False,
        num_workers=config["data"].get("num_workers", 0),
        root=root,
    )

    model = build_router(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    use_amp = config["training"].get("amp", True) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    if smoke_test:
        phases = [(1, 1)]
        phase1_lr = config["training"]["phase1_lr"]
        phase2_lr = config["training"]["phase2_lr"]
    else:
        phases = [
            (1, config["training"]["phase1_epochs"]),
            (2, config["training"]["phase2_epochs"]),
        ]
        phase1_lr = config["training"]["phase1_lr"]
        phase2_lr = config["training"]["phase2_lr"]

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_path = CHECKPOINTS_DIR / config["checkpoint"]["filename"]

    early_cfg = config["training"].get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", False) and not smoke_test
    early_patience = early_cfg.get("patience", 2)
    early_min_delta = early_cfg.get("min_delta", 0.0005)
    epochs_without_improvement = 0
    stop_training = False

    for phase, epochs in phases:
        if stop_training:
            break
        set_router_train_phase(
            model,
            phase=phase,
            unfreeze_blocks=config["training"].get("unfreeze_blocks", 2),
        )
        lr = phase1_lr if phase == 1 else phase2_lr
        optimizer = AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

        print(f"--- Router phase {phase}: up to {epochs} epoch(s), lr={lr} ---")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, use_amp, scaler
            )
            val_metrics = evaluate_router(model, val_loader, device)
            scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"leaf_p/r={val_metrics['leaf']['precision']:.3f}/"
                f"{val_metrics['leaf']['recall']:.3f} "
                f"fruit_p/r={val_metrics['fruit']['precision']:.3f}/"
                f"{val_metrics['fruit']['recall']:.3f}"
            )

            improved_for_checkpoint = val_metrics["accuracy"] > best_val_acc
            improved_for_early_stop = (
                val_metrics["accuracy"] > best_val_acc + early_min_delta
            )

            if improved_for_checkpoint:
                best_val_acc = val_metrics["accuracy"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "val_metrics": val_metrics,
                        "phase": phase,
                        "epoch": epoch,
                    },
                    best_path,
                )

            if improved_for_early_stop or not early_enabled:
                epochs_without_improvement = 0
            elif early_enabled:
                epochs_without_improvement += 1

            if early_enabled and epochs_without_improvement >= early_patience:
                print(
                    f"Early stopping: val_acc did not improve by >{early_min_delta} "
                    f"for {early_patience} consecutive epoch(s)."
                )
                stop_training = True
                break

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {best_path}")
    return best_path


def _early_stop_score(metrics: dict, metric_name: str) -> float:
    if metric_name == "macro_f1":
        return metrics["macro_f1"]
    return metrics["accuracy"]


def _save_leaf_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict,
    class_names: list[str],
    val_metrics: dict,
    phase: int,
    epoch: int,
    metrics_history: list[dict],
    best_macro_f1: float,
    completed_phases: list[int],
    state_dict: dict | None = None,
) -> None:
    """Save checkpoint. ``state_dict`` defaults to current model (must be best weights)."""
    payload = {
        "model_state_dict": state_dict if state_dict is not None else model.state_dict(),
        "config": config,
        "class_names": class_names,
        "val_metrics": val_metrics,
        "phase": phase,
        "epoch": epoch,
        "best_phase": phase,
        "best_epoch": epoch,
        "metrics_history": metrics_history,
        "best_macro_f1": best_macro_f1,
        "completed_phases": completed_phases,
    }
    torch.save(payload, path)


def _late_blight_fp_to_early_blight(val_metrics: dict) -> int:
    class_names = val_metrics["class_names"]
    confusion = val_metrics["confusion"]
    eb_idx = class_names.index("Early blight")
    lb_idx = class_names.index("Late blight")
    return int(confusion[lb_idx][eb_idx])


def train_leaf(
    config: dict,
    *,
    start_phase: int = 1,
    max_phase: int = 2,
    resume_from: Path | None = None,
    extra_phase1_epochs: int = 0,
    disable_early_stopping: bool = False,
    overwrite: bool = False,
) -> Path:
    guard_checkpoint_path(CHECKPOINTS_DIR / config["checkpoint"]["filename"], overwrite)
    set_seed(config["training"]["seed"])
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Training device: {device}")

    root = repo_root()
    batch_size = config["data"]["batch_size"]
    image_size = config["data"]["image_size"]
    class_names, _ = load_class_map(root / "data" / "splits" / "leaf")

    train_loader = build_leaf_dataloader(
        "train",
        batch_size=batch_size,
        image_size=image_size,
        num_workers=config["data"].get("num_workers", 0),
        root=root,
    )
    val_loader = build_leaf_dataloader(
        "val",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        root=root,
    )

    class_weights = compute_class_weights(root).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = config["training"].get("amp", True) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINTS_DIR / config["checkpoint"]["filename"]
    metrics_history: list[dict] = []
    completed_phases: list[int] = []
    best_macro_f1 = 0.0
    best_state_dict: dict | None = None
    best_phase = 0
    best_epoch = 0
    best_val_metrics: dict | None = None

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model = build_leaf_classifier(num_classes=len(class_names), pretrained=False).to(
            device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics_history = list(checkpoint.get("metrics_history", []))
        completed_phases = list(checkpoint.get("completed_phases", []))
        best_macro_f1 = float(checkpoint.get("best_macro_f1", 0.0))
        best_state_dict = checkpoint["model_state_dict"]
        best_phase = int(checkpoint.get("best_phase", checkpoint.get("phase", 0)))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        best_val_metrics = checkpoint.get("val_metrics")
        print(
            f"Resumed from {resume_from} "
            f"(best macro-F1={best_macro_f1:.4f} at phase {best_phase} epoch {best_epoch})"
        )
    else:
        model = build_leaf_classifier(num_classes=len(class_names), pretrained=True).to(
            device
        )

    early_cfg = config["training"].get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", True) and not disable_early_stopping
    if disable_early_stopping:
        print("Early stopping disabled for this run.")
    early_patience = early_cfg.get("patience", 4)
    early_min_delta = early_cfg.get("min_delta", 0.003)
    early_metric = early_cfg.get("metric", "macro_f1")

    phase_specs = [
        (1, config["training"]["phase1_epochs"], config["training"]["phase1_lr"]),
        (2, config["training"]["phase2_epochs"], config["training"]["phase2_lr"]),
    ]
    if extra_phase1_epochs > 0:
        if resume_from is None:
            raise ValueError("--leaf-extra-phase1-epochs requires --resume checkpoint.")
        phase_specs = [(1, extra_phase1_epochs, config["training"]["phase1_lr"])]
        start_phase = 1
        max_phase = 1
        early_enabled = False
        print(
            f"Phase 1 extension: {extra_phase1_epochs} additional head-only epoch(s), "
            "early stopping disabled."
        )

    premature_early_stop = False
    stop_training = False

    for phase, epochs, lr in phase_specs:
        if phase < start_phase or phase > max_phase:
            continue
        if stop_training:
            break

        set_leaf_train_phase(
            model,
            phase=phase,
            unfreeze_blocks=config["training"].get("unfreeze_blocks", 3),
        )
        optimizer = AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        epochs_without_improvement = 0

        print(f"--- Leaf phase {phase}: up to {epochs} epoch(s), lr={lr} ---")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, use_amp, scaler
            )
            val_metrics = evaluate_leaf(model, val_loader, device, class_names)
            scheduler.step()

            score = _early_stop_score(val_metrics, early_metric)
            metrics_history.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "global_epoch": len(metrics_history) + 1,
                    "macro_f1": val_metrics["macro_f1"],
                    "accuracy": val_metrics["accuracy"],
                    "weighted_f1": val_metrics["weighted_f1"],
                    "early_blight_precision": val_metrics["per_class"]["Early blight"][
                        "precision"
                    ],
                    "late_blight_fp_to_early_blight": _late_blight_fp_to_early_blight(
                        val_metrics
                    ),
                }
            )

            mosaic_recall = val_metrics["per_class"]["Tomato mosaic virus"]["recall"]
            leaf_mold_recall = val_metrics["per_class"]["Leaf Mold"]["recall"]
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"mosaic_r={mosaic_recall:.3f} leaf_mold_r={leaf_mold_recall:.3f}"
            )

            improved_for_early_stop = score > best_macro_f1 + early_min_delta

            if score > best_macro_f1:
                best_macro_f1 = score
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_phase = phase
                best_epoch = epoch
                best_val_metrics = val_metrics
                _save_leaf_checkpoint(
                    best_path,
                    model,
                    config,
                    class_names,
                    best_val_metrics,
                    best_phase,
                    best_epoch,
                    metrics_history,
                    best_macro_f1,
                    completed_phases,
                    state_dict=best_state_dict,
                )

            if improved_for_early_stop:
                epochs_without_improvement = 0
            elif early_enabled:
                epochs_without_improvement += 1

            if early_enabled and epochs_without_improvement >= early_patience:
                rising = macro_f1_still_rising(metrics_history)
                print(
                    f"Early stopping: val {early_metric} did not improve by "
                    f">{early_min_delta} for {early_patience} consecutive epoch(s)."
                )
                if rising:
                    premature_early_stop = True
                    print(
                        "FLAG: macro-F1 still trending upward over the last 3 epochs — "
                        "early stopping may be too aggressive (consider loosening "
                        "patience/min_delta and rerunning)."
                    )
                stop_training = True
                break

        if phase not in completed_phases:
            completed_phases.append(phase)
        if best_state_dict is not None and best_val_metrics is not None:
            _save_leaf_checkpoint(
                best_path,
                model,
                config,
                class_names,
                best_val_metrics,
                best_phase,
                best_epoch,
                metrics_history,
                best_macro_f1,
                completed_phases,
                state_dict=best_state_dict,
            )

        if phase == 1 and max_phase == 1 and extra_phase1_epochs == 0:
            print()
            if best_val_metrics is not None:
                print_leaf_metrics(best_val_metrics, "val (best checkpoint)")
            else:
                print_leaf_metrics(val_metrics, "val (end of phase 1)")
            gate_cfg = config.get("phase1_gate", {})
            flagged, message = check_phase1_minority_recall(
                best_val_metrics if best_val_metrics is not None else val_metrics,
                weak_margin=gate_cfg.get("weak_recall_margin", 0.10),
                min_absolute_recall=gate_cfg.get("min_absolute_recall", 0.70),
            )
            print(message)
            if flagged:
                print(
                    "STOP: review before phase 2 — consider weighted sampling addition."
                )

    print()
    print_macro_f1_trajectory(metrics_history)
    print()
    print(f"Best val macro-F1: {best_macro_f1:.4f}")
    print(f"Checkpoint saved to: {best_path}")
    if premature_early_stop:
        print(
            "NOTE: premature early-stop flag was raised — review trajectory before "
            "accepting checkpoint as final."
        )
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train router or leaf classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "router.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 1-epoch phase-1 GPU smoke test (router only).",
    )
    parser.add_argument(
        "--leaf-max-phase",
        type=int,
        choices=[1, 2],
        default=2,
        help="Stop after this leaf training phase (default: 2).",
    )
    parser.add_argument(
        "--leaf-start-phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Leaf phase to start from (default: 1).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume leaf training from checkpoint.",
    )
    parser.add_argument(
        "--leaf-extra-phase1-epochs",
        type=int,
        default=0,
        help="Continue phase 1 head-only for N extra epochs from --resume.",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping for leaf training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing checkpoint file (off by default).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if config.get("model") == "router":
        train_router(config, smoke_test=args.smoke_test, overwrite=args.overwrite)
    elif config.get("model") == "leaf":
        train_leaf(
            config,
            start_phase=args.leaf_start_phase,
            max_phase=args.leaf_max_phase,
            resume_from=args.resume,
            extra_phase1_epochs=args.leaf_extra_phase1_epochs,
            disable_early_stopping=args.no_early_stopping,
            overwrite=args.overwrite,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.get('model')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
