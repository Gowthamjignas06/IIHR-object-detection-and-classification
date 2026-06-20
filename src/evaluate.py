"""Evaluation utilities for router and leaf classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

DOMAIN_NAMES = ("leaf", "fruit")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@torch.no_grad()
def evaluate_router(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    model.eval()
    confusion = torch.zeros(2, 2, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
            confusion[true_label, pred_label] += 1

    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    metrics: dict = {
        "accuracy": _safe_div(correct, total),
        "confusion": confusion.cpu().tolist(),
        "support": {},
        "leaf": {},
        "fruit": {},
    }

    for idx, name in enumerate(DOMAIN_NAMES):
        tp = int(confusion[idx, idx].item())
        fp = int(confusion[1 - idx, idx].item())
        fn = int(confusion[idx, 1 - idx].item())
        support = int(confusion[idx].sum().item())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        metrics["support"][name] = support

    return metrics


def metrics_from_confusion(
    confusion: torch.Tensor,
    class_names: list[str],
) -> dict:
    num_classes = len(class_names)
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())

    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    weighted_f1_sum = 0.0
    support_total = 0

    for idx, name in enumerate(class_names):
        tp = int(confusion[idx, idx].item())
        fp = int(confusion[:, idx].sum().item()) - tp
        fn = int(confusion[idx, :].sum().item()) - tp
        support = int(confusion[idx].sum().item())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)
        weighted_f1_sum += f1 * support
        support_total += support

    macro_f1 = _safe_div(sum(f1_scores), len(f1_scores))
    weighted_f1 = _safe_div(weighted_f1_sum, support_total)

    return {
        "accuracy": _safe_div(correct, total),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion": confusion.cpu().tolist(),
        "per_class": per_class,
        "class_names": class_names,
    }


@torch.no_grad()
def evaluate_leaf(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
            confusion[true_label, pred_label] += 1

    return metrics_from_confusion(confusion, class_names)


def print_leaf_metrics(metrics: dict, split_name: str = "val") -> None:
    print(f"=== Leaf classifier metrics ({split_name}) ===")
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro-F1: {metrics['macro_f1']:.4f} | "
        f"Weighted-F1: {metrics['weighted_f1']:.4f}"
    )
    print("Confusion matrix [true x pred]:")
    for row in metrics["confusion"]:
        print(f"  {row}")
    print(f"{'Class':<40} {'Prec':>7} {'Recall':>7} {'F1':>7} {'N':>6}")
    print("-" * 72)
    for name in metrics["class_names"]:
        stats = metrics["per_class"][name]
        print(
            f"{name:<40} {stats['precision']:>7.4f} "
            f"{stats['recall']:>7.4f} {stats['f1']:>7.4f} "
            f"{stats['support']:>6}"
        )


def print_macro_f1_trajectory(history: list[dict]) -> None:
    print("=== Val macro-F1 trajectory ===")
    has_global = any("global_epoch" in row for row in history)
    if has_global:
        print(
            f"{'Phase':>5} {'Epoch':>5} {'Global':>6} {'Macro-F1':>9} "
            f"{'EB-Prec':>8} {'Accuracy':>9}"
        )
        print("-" * 50)
        for row in history:
            eb_prec = row.get("early_blight_precision", float("nan"))
            print(
                f"{row['phase']:>5} {row['epoch']:>5} {row.get('global_epoch', '-'):>6} "
                f"{row['macro_f1']:>9.4f} {eb_prec:>8.4f} {row['accuracy']:>9.4f}"
            )
    else:
        print(f"{'Phase':>5} {'Epoch':>5} {'Macro-F1':>9} {'Accuracy':>9}")
        print("-" * 32)
        for row in history:
            print(
                f"{row['phase']:>5} {row['epoch']:>5} "
                f"{row['macro_f1']:>9.4f} {row['accuracy']:>9.4f}"
            )


def macro_f1_still_rising(history: list[dict], window: int = 3) -> bool:
    if len(history) < window:
        return False
    tail = [row["macro_f1"] for row in history[-window:]]
    return all(tail[i] < tail[i + 1] for i in range(len(tail) - 1))


def check_phase1_minority_recall(
    metrics: dict,
    *,
    weak_margin: float = 0.10,
    min_absolute_recall: float = 0.70,
) -> tuple[bool, str]:
    watch_classes = ("Tomato mosaic virus", "Leaf Mold")
    recalls = {
        name: metrics["per_class"][name]["recall"]
        for name in metrics["class_names"]
        if name not in watch_classes
    }
    median_others = sorted(recalls.values())[len(recalls) // 2]
    messages = []
    flagged = False
    for name in watch_classes:
        recall = metrics["per_class"][name]["recall"]
        if recall < min_absolute_recall or recall < median_others - weak_margin:
            flagged = True
            messages.append(
                f"{name}: recall={recall:.4f} "
                f"(median of other 8={median_others:.4f}, "
                f"margin={weak_margin}, min={min_absolute_recall})"
            )
    if flagged:
        return True, "WEAK minority recall detected — " + "; ".join(messages)
    return False, (
        "Phase 1 gate passed: Tomato mosaic virus and Leaf Mold recall "
        "are in line with other classes."
    )


def load_leaf_checkpoint(checkpoint_path: Path, device: torch.device):
    from models.leaf_classifier import build_leaf_classifier

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    class_names = checkpoint["class_names"]
    model = build_leaf_classifier(num_classes=len(class_names), pretrained=False).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_leaf_checkpoint(
    checkpoint_path: Path,
    splits: tuple[str, ...] = ("val", "test"),
    batch_size: int = 16,
    image_size: int = 224,
) -> dict[str, dict]:
    from datasets import build_leaf_dataloader, load_class_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_leaf_checkpoint(checkpoint_path, device)
    class_names = checkpoint["class_names"]
    root = repo_root()
    results: dict[str, dict] = {}
    for split in splits:
        loader = build_leaf_dataloader(
            split,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
            root=root,
        )
        results[split] = evaluate_leaf(model, loader, device, class_names)
    return results


def print_router_metrics(metrics: dict, split_name: str = "val") -> None:
    print(f"=== Router metrics ({split_name}) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Confusion matrix [true x pred] (leaf=0, fruit=1):")
    for row in metrics["confusion"]:
        print(f"  {row}")
    for name in DOMAIN_NAMES:
        stats = metrics[name]
        support = metrics["support"][name]
        print(
            f"{name:>5}: precision={stats['precision']:.4f} "
            f"recall={stats['recall']:.4f} f1={stats['f1']:.4f} "
            f"support={support}"
        )


def print_fruit_val_test_comparison(val_metrics: dict, test_metrics: dict) -> None:
    print("=== Fruit-side val vs test comparison ===")
    print(f"{'Metric':<12} {'Val':>8} {'Test':>8} {'Delta':>8}")
    for key in ("precision", "recall", "f1"):
        val_value = val_metrics["fruit"][key]
        test_value = test_metrics["fruit"][key]
        delta = test_value - val_value
        print(f"{key:<12} {val_value:>8.4f} {test_value:>8.4f} {delta:>+8.4f}")
    print(
        f"{'support':<12} {val_metrics['support']['fruit']:>8} "
        f"{test_metrics['support']['fruit']:>8}"
    )


def fruit_test_gap_flagged(
    val_metrics: dict,
    test_metrics: dict,
    threshold: float = 0.01,
) -> tuple[bool, str]:
    drops = []
    for key in ("precision", "recall", "f1"):
        delta = test_metrics["fruit"][key] - val_metrics["fruit"][key]
        if delta < -threshold:
            drops.append(f"{key} dropped {abs(delta):.4f} on test vs val")
    if drops:
        return True, "; ".join(drops)
    return False, "Fruit-side test metrics are within threshold of val."


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_router_checkpoint(checkpoint_path: Path, device: torch.device):
    from models.router import build_router

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = build_router(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate_router_checkpoint(
    checkpoint_path: Path,
    splits: tuple[str, ...] = ("val", "test"),
    batch_size: int = 16,
    image_size: int = 224,
) -> dict[str, dict]:
    from datasets import build_router_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_router_checkpoint(checkpoint_path, device)
    root = repo_root()
    results: dict[str, dict] = {}
    for split in splits:
        loader = build_router_dataloader(
            split,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
            equal_domain_sampling=False,
            root=root,
        )
        results[split] = evaluate_router(model, loader, device)
    return results


def build_parser() -> argparse.ArgumentParser:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model checkpoint.")
    parser.add_argument(
        "--model",
        choices=["router", "leaf"],
        default="router",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.checkpoint is None:
        if args.model == "leaf":
            args.checkpoint = repo_root() / "checkpoints" / "leaf_v1_efficientnet_b0.pt"
        else:
            args.checkpoint = repo_root() / "checkpoints" / "router_v1_mobilenet_v3_small.pt"

    if args.model == "leaf":
        results = evaluate_leaf_checkpoint(args.checkpoint, tuple(args.splits))
        for split in args.splits:
            print_leaf_metrics(results[split], split)
            print()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        history = checkpoint.get("metrics_history", [])
        if history:
            print_macro_f1_trajectory(history)
            print()
        return 0

    results = evaluate_router_checkpoint(args.checkpoint, tuple(args.splits))
    for split in args.splits:
        print_router_metrics(results[split], split)
        print()
    if "val" in results and "test" in results:
        print_fruit_val_test_comparison(results["val"], results["test"])
        flagged, message = fruit_test_gap_flagged(results["val"], results["test"])
        print()
        print(message)
        if flagged:
            print("FLAG: meaningful fruit-side test drop detected — review before step 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
