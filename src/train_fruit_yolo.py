"""Thin wrapper around Ultralytics YOLO training for the fruit detector."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def repo_root() -> Path:
    return REPO_ROOT


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_name: str) -> str | int:
    if device_name == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device_name in {"0", "cuda", "cuda:0"}:
        return 0
    return device_name


def resolve_data_yaml(config: dict) -> Path:
    data_yaml = Path(config["data"]["yaml"])
    if not data_yaml.is_absolute():
        data_yaml = repo_root() / data_yaml
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing fruit data.yaml: {data_yaml}")
    return data_yaml.resolve()


def write_resolved_data_yaml(source_yaml: Path) -> Path:
    """Write a data.yaml with an absolute dataset root path for Ultralytics."""
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    payload["path"] = str(source_yaml.parent.resolve())
    resolved = repo_root() / "runs" / "fruit" / "_data_resolved.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return resolved


def build_train_kwargs(
    config: dict,
    *,
    epochs: int,
    run_name: str,
    smoke_test: bool = False,
    batch: int | None = None,
) -> dict:
    training = config["training"]
    output = config["output"]
    project = Path(output["project"])
    if not project.is_absolute():
        project = repo_root() / project

    batch_size = batch if batch is not None else int(training["batch"])
    return {
        "data": str(write_resolved_data_yaml(resolve_data_yaml(config))),
        "epochs": epochs,
        "imgsz": int(training["imgsz"]),
        "batch": batch_size,
        "patience": int(training["patience"]),
        "seed": int(training["seed"]),
        "device": resolve_device(training.get("device", "auto")),
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
        "workers": int(training.get("workers", 0)),
        "amp": bool(training.get("amp", True)),
        "pretrained": True,
        "verbose": True,
        "val": True,
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message


def train_fruit_yolo(
    config: dict,
    *,
    smoke_test: bool = False,
    epochs: int | None = None,
    overwrite: bool = False,
) -> Path:
    training = config["training"]
    output = config["output"]
    weights_name = training["pretrained_weights"]
    run_name = "smoke_test" if smoke_test else output.get("run_name", "train")
    epoch_count = 1 if smoke_test else int(epochs or training["epochs"])

    dest = CHECKPOINTS_DIR / config["checkpoint"]["filename"]
    if not smoke_test and dest.exists() and not overwrite:
        # Checked before training so a long run can't end by clobbering v1.
        raise FileExistsError(
            f"{dest} already exists. Set checkpoint.filename in the config to a new "
            "name (e.g. fruit_v2_yolov8n.pt) or pass --overwrite."
        )

    device = resolve_device(training.get("device", "auto"))
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_yaml = resolve_data_yaml(config)
    print(f"Data: {data_yaml}")
    print(f"Resolved Ultralytics data root: {data_yaml.parent}")
    print(f"Run: {'smoke test' if smoke_test else 'full training'} | epochs={epoch_count}")

    model = YOLO(weights_name)
    train_kwargs = build_train_kwargs(
        config,
        epochs=epoch_count,
        run_name=run_name,
        smoke_test=smoke_test,
    )

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as exc:
        fallback = int(training.get("batch_fallback", 4))
        primary = int(training["batch"])
        if _is_cuda_oom(exc) and train_kwargs["batch"] > fallback:
            print(
                f"CUDA OOM at batch={primary}. Retrying with batch={fallback}."
            )
            train_kwargs["batch"] = fallback
            model = YOLO(weights_name)
            results = model.train(**train_kwargs)
        else:
            raise

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    chosen = best_weights if best_weights.exists() else last_weights

    if not smoke_test and chosen.exists():
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chosen, dest)
        print(f"Checkpoint copied to: {dest}")

    print(f"Ultralytics run dir: {save_dir}")
    print(f"Weights: {chosen}")
    return save_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fruit YOLO detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "fruit_yolo.yaml",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 1-epoch GPU smoke test.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config epoch count (ignored with --smoke-test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing checkpoint file (off by default).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    train_fruit_yolo(
        config,
        smoke_test=args.smoke_test,
        epochs=args.epochs,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
