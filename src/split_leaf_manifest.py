"""Create stratified train/val/test CSV manifests for the leaf dataset.

Reads images from ``data/leaf_raw/PlantVillage/<class>/`` (see AUDIT.md) and
writes ``image_path,label`` CSVs under ``data/splits/leaf/``.

Usage:
    python src/split_leaf_manifest.py --dry-run
    python src/split_leaf_manifest.py --write
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")

# PlantVillage folder name -> architecture human-readable class label
FOLDER_TO_LABEL: dict[str, str] = {
    "Tomato_Bacterial_spot": "Bacterial spot",
    "Tomato_Early_blight": "Early blight",
    "Tomato_Late_blight": "Late blight",
    "Tomato_Leaf_Mold": "Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider mites (Two-spotted)",
    "Tomato__Target_Spot": "Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato mosaic virus",
    "Tomato_healthy": "Healthy",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def leaf_data_root(leaf_raw: Path) -> Path:
    nested = leaf_raw / "PlantVillage"
    if nested.is_dir():
        return nested
    return leaf_raw


def collect_images(leaf_raw: Path) -> dict[str, list[Path]]:
    root = leaf_data_root(leaf_raw)
    by_class: dict[str, list[Path]] = {}

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        folder_name = class_dir.name
        if folder_name not in FOLDER_TO_LABEL:
            raise ValueError(
                f"Unexpected class folder {folder_name!r} under {root}. "
                f"Update FOLDER_TO_LABEL in split_leaf_manifest.py."
            )

        label = FOLDER_TO_LABEL[folder_name]
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No images found in {class_dir}")
        by_class[label] = paths

    expected = set(FOLDER_TO_LABEL.values())
    found = set(by_class)
    if found != expected:
        missing = expected - found
        extra = found - expected
        raise ValueError(f"Class mismatch. Missing={missing!r}, extra={extra!r}")

    return by_class


def stratified_image_split(
    by_class: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, str]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    rng = random.Random(seed)
    assigned: dict[str, list[tuple[Path, str]]] = {name: [] for name in SPLIT_NAMES}

    for label in sorted(by_class):
        paths = list(by_class[label])
        rng.shuffle(paths)
        n = len(paths)
        if n == 0:
            continue

        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        n_test = n - n_train - n_val

        # Classes with n < 3 skip the floor below and may get 0 val or test images.
        if n >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)
            while n_train + n_val + n_test > n:
                if n_train >= n_val and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            while n_train + n_val + n_test < n:
                n_train += 1

        assigned["train"].extend((path, label) for path in paths[:n_train])
        assigned["val"].extend((path, label) for path in paths[n_train : n_train + n_val])
        assigned["test"].extend((path, label) for path in paths[n_train + n_val :])

    return assigned


def summarize_split(
    by_class: dict[str, list[Path]],
    split_assignments: dict[str, list[tuple[Path, str]]],
) -> dict:
    total_images = sum(len(paths) for paths in by_class.values())
    summary: dict = {
        "totals": {
            "classes": len(by_class),
            "images": total_images,
        },
        "images_by_class_all": {
            label: len(paths) for label, paths in sorted(by_class.items())
        },
        "splits": {},
    }

    for split_name in SPLIT_NAMES:
        samples = split_assignments[split_name]
        class_counts = Counter(label for _, label in samples)
        summary["splits"][split_name] = {
            "images": len(samples),
            "image_fraction": round(len(samples) / total_images, 4) if total_images else 0,
            "images_by_class": dict(
                sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }

    return summary


def print_summary(summary: dict) -> None:
    totals = summary["totals"]
    print("=== Leaf stratified split preview ===")
    print(f"Classes: {totals['classes']}")
    print(f"Total images: {totals['images']}")
    print()
    print(f"{'Split':<6} {'Images':>7} {'Img%':>7}")
    for split_name in SPLIT_NAMES:
        split = summary["splits"][split_name]
        print(f"{split_name:<6} {split['images']:>7} {split['image_fraction']:>7.1%}")
    print()
    all_classes = sorted(summary["images_by_class_all"])
    print(f"{'Class':<40} {'All':>6} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 72)
    for label in all_classes:
        all_count = summary["images_by_class_all"][label]
        train_count = summary["splits"]["train"]["images_by_class"].get(label, 0)
        val_count = summary["splits"]["val"]["images_by_class"].get(label, 0)
        test_count = summary["splits"]["test"]["images_by_class"].get(label, 0)
        print(
            f"{label:<40} {all_count:>6} {train_count:>7} {val_count:>7} {test_count:>7}"
        )


def relative_image_path(image_path: Path, root: Path) -> str:
    resolved = image_path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_manifests(
    split_assignments: dict[str, list[tuple[Path, str]]],
    output_dir: Path,
    root: Path,
    summary: dict,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted(FOLDER_TO_LABEL.values())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    (output_dir / "class_map.json").write_text(
        json.dumps({"class_names": class_names, "class_to_idx": class_to_idx}, indent=2),
        encoding="utf-8",
    )

    for split_name in SPLIT_NAMES:
        manifest_path = output_dir / f"{split_name}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "label"])
            for image_path, label in sorted(
                split_assignments[split_name], key=lambda item: (item[1], str(item[0]))
            ):
                writer.writerow([relative_image_path(image_path, root), label])

    payload = {
        **summary,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "leaf_data_root": "data/leaf_raw/PlantVillage",
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split leaf dataset into CSV manifests.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Preview split only.")
    mode.add_argument("--write", action="store_true", help="Write CSV manifests.")
    parser.add_argument("--leaf-raw", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview-json",
        type=Path,
        default=None,
        help="Dry-run JSON path (default: data/splits/leaf/dry_run_preview.json).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    leaf_raw = (args.leaf_raw or root / "data" / "leaf_raw").resolve()
    output_dir = (args.output_dir or root / "data" / "splits" / "leaf").resolve()
    preview_json = (
        args.preview_json or output_dir / "dry_run_preview.json"
    ).resolve()

    by_class = collect_images(leaf_raw)
    split_assignments = stratified_image_split(
        by_class,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    summary = summarize_split(by_class, split_assignments)
    print_summary(summary)

    preview_payload = {
        **summary,
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "mode": "dry-run" if args.dry_run else "write",
    }
    preview_json.parent.mkdir(parents=True, exist_ok=True)
    preview_json.write_text(json.dumps(preview_payload, indent=2), encoding="utf-8")
    print()
    print(f"Preview JSON saved to: {preview_json}")

    if args.write:
        write_manifests(
            split_assignments,
            output_dir,
            root,
            summary,
            seed=args.seed,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
        )
        print(f"Leaf manifests written to: {output_dir}")
    else:
        print("Dry run only — no CSVs written. Re-run with --write after review.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
