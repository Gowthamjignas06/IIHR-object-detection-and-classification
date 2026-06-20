"""Group Roboflow fruit images by source identity and split at group level.

Roboflow exports 3 augmented copies of many source images. Splitting individual
files across train/val/test leaks augmented copies and inflates metrics. This
script groups by the filename stem before ``.rf.`` and assigns whole groups
to a split.

Usage:
    python src/split_fruit_groups.py --dry-run
    python src/split_fruit_groups.py --write
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "val", "test")
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class GroupMember:
    image_path: Path
    label_path: Path
    roboflow_split: str


@dataclass
class SourceGroup:
    key: str
    members: list[GroupMember] = field(default_factory=list)
    primary_class_id: int = -1
    primary_class_name: str = ""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_group_key(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def load_class_names(fruit_raw: Path) -> list[str]:
    data_yaml = fruit_raw / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")

    names: list[str] = []
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("names:"):
            raw = line.split(":", 1)[1].strip()
            if raw.startswith("[") and raw.endswith("]"):
                import ast

                names = list(ast.literal_eval(raw))
            break

    if not names:
        raise ValueError(f"Could not parse class names from {data_yaml}")
    return names


def parse_label_classes(label_path: Path) -> list[int]:
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed label line in {label_path}: {line!r}")
        classes.append(int(float(parts[0])))
    return classes


def primary_class_from_label(label_path: Path) -> int:
    classes = parse_label_classes(label_path)
    if not classes:
        return -1
    return Counter(classes).most_common(1)[0][0]


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_groups(fruit_raw: Path, class_names: list[str]) -> dict[str, SourceGroup]:
    groups: dict[str, SourceGroup] = {}

    for roboflow_split in ROBOFLOW_SPLITS:
        images_dir = fruit_raw / roboflow_split / "images"
        labels_dir = fruit_raw / roboflow_split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {roboflow_split}/images and {roboflow_split}/labels under {fruit_raw}"
            )

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image_for_label(label_path, images_dir)
            if image_path is None:
                raise FileNotFoundError(
                    f"No matching image for label {label_path} in {images_dir}"
                )

            key = source_group_key(label_path.name)
            if key not in groups:
                groups[key] = SourceGroup(key=key)
            groups[key].members.append(
                GroupMember(
                    image_path=image_path,
                    label_path=label_path,
                    roboflow_split=roboflow_split,
                )
            )

    for group in groups.values():
        class_ids = [
            primary_class_from_label(member.label_path)
            for member in group.members
            if member.label_path.exists()
        ]
        class_ids = [cid for cid in class_ids if cid >= 0]
        if not class_ids:
            group.primary_class_id = -1
            group.primary_class_name = "unknown"
            continue
        primary_id = Counter(class_ids).most_common(1)[0][0]
        group.primary_class_id = primary_id
        group.primary_class_name = class_names[primary_id]

    return groups


def group_size_histogram(groups: dict[str, SourceGroup]) -> dict[str, int]:
    sizes = Counter(len(group.members) for group in groups.values())
    return {str(size): count for size, count in sorted(sizes.items())}


def groups_by_primary_class(groups: dict[str, SourceGroup]) -> dict[str, int]:
    counts = Counter(group.primary_class_name for group in groups.values())
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def stratified_group_split(
    groups: dict[str, SourceGroup],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    by_class: dict[str, list[str]] = defaultdict(list)
    for key, group in groups.items():
        by_class[group.primary_class_name].append(key)

    for class_name in by_class:
        by_class[class_name].sort()

    rng = random.Random(seed)
    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    for class_name in sorted(by_class):
        keys = by_class[class_name]
        rng.shuffle(keys)
        n = len(keys)
        if n == 0:
            continue

        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        n_test = n - n_train - n_val

        # Classes with n < 3 groups skip the floor below and may get 0 val or test groups.
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

        assigned["train"].extend(keys[:n_train])
        assigned["val"].extend(keys[n_train : n_train + n_val])
        assigned["test"].extend(keys[n_train + n_val :])

    return assigned


def summarize_split(
    groups: dict[str, SourceGroup],
    split_assignments: dict[str, list[str]],
    class_names: list[str],
) -> dict:
    summary: dict = {
        "totals": {},
        "group_size_histogram": group_size_histogram(groups),
        "groups_by_primary_class": groups_by_primary_class(groups),
        "splits": {},
    }

    total_groups = len(groups)
    total_images = sum(len(g.members) for g in groups.values())
    summary["totals"] = {
        "groups": total_groups,
        "images": total_images,
    }

    for split_name in SPLIT_NAMES:
        keys = split_assignments[split_name]
        split_groups = [groups[key] for key in keys]
        images = sum(len(g.members) for g in split_groups)
        class_group_counts = Counter(g.primary_class_name for g in split_groups)
        class_image_counts: Counter[str] = Counter()
        for group in split_groups:
            class_image_counts[group.primary_class_name] += len(group.members)

        summary["splits"][split_name] = {
            "groups": len(keys),
            "images": images,
            "group_fraction": round(len(keys) / total_groups, 4) if total_groups else 0,
            "image_fraction": round(images / total_images, 4) if total_images else 0,
            "groups_by_primary_class": dict(
                sorted(class_group_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "images_by_primary_class": dict(
                sorted(class_image_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }

    summary["class_names"] = class_names
    return summary


def print_summary(summary: dict) -> None:
    totals = summary["totals"]
    print("=== Fruit source-group split preview ===")
    print(f"Total source groups: {totals['groups']}")
    print(f"Total images: {totals['images']}")
    print(f"Group size histogram: {summary['group_size_histogram']}")
    print()
    print("Groups by primary class (all data):")
    for name, count in summary["groups_by_primary_class"].items():
        print(f"  {name}: {count}")
    print()
    print(f"{'Split':<6} {'Groups':>7} {'Images':>7} {'Grp%':>7} {'Img%':>7}")
    for split_name in SPLIT_NAMES:
        split = summary["splits"][split_name]
        print(
            f"{split_name:<6} {split['groups']:>7} {split['images']:>7} "
            f"{split['group_fraction']:>7.1%} {split['image_fraction']:>7.1%}"
        )
    print()
    for split_name in SPLIT_NAMES:
        split = summary["splits"][split_name]
        print(f"--- {split_name}: groups by primary class ---")
        for name, count in split["groups_by_primary_class"].items():
            img_count = split["images_by_primary_class"].get(name, 0)
            print(f"  {name}: {count} groups, {img_count} images")


def write_split_outputs(
    groups: dict[str, SourceGroup],
    split_assignments: dict[str, list[str]],
    output_dir: Path,
    class_names: list[str],
    summary: dict,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in OUTPUT_SPLITS:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_key",
                "split",
                "primary_class",
                "primary_class_id",
                "member_count",
                "image_filename",
                "label_filename",
                "roboflow_split",
            ]
        )
        for split_name in SPLIT_NAMES:
            for group_key in sorted(split_assignments[split_name]):
                group = groups[group_key]
                for member in sorted(group.members, key=lambda m: m.image_path.name):
                    dest_image = output_dir / "images" / split_name / member.image_path.name
                    dest_label = output_dir / "labels" / split_name / member.label_path.name
                    shutil.copy2(member.image_path, dest_image)
                    shutil.copy2(member.label_path, dest_label)
                    writer.writerow(
                        [
                            group.key,
                            split_name,
                            group.primary_class_name,
                            group.primary_class_id,
                            len(group.members),
                            member.image_path.name,
                            member.label_path.name,
                            member.roboflow_split,
                        ]
                    )

    data_yaml_path = output_dir / "data.yaml"
    names_yaml = ", ".join(repr(name) for name in class_names)
    data_yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                f"nc: {len(class_names)}",
                f"names: [{names_yaml}]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary_payload = {
        **summary,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "grouping_key": "filename stem before .rf.",
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Roboflow fruit dataset by source-image groups."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview grouping and split sizes without copying files.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Materialize YOLO folder layout under data/splits/fruit/.",
    )
    parser.add_argument(
        "--fruit-raw",
        type=Path,
        default=None,
        help="Path to Roboflow export (default: data/fruit_raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for --write (default: data/splits/fruit).",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train group ratio.")
    parser.add_argument("--val", type=float, default=0.15, help="Val group ratio.")
    parser.add_argument("--test", type=float, default=0.15, help="Test group ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--preview-json",
        type=Path,
        default=None,
        help="Where to save dry-run JSON (default: data/splits/fruit/dry_run_preview.json).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root()
    fruit_raw = (args.fruit_raw or root / "data" / "fruit_raw").resolve()
    output_dir = (args.output_dir or root / "data" / "splits" / "fruit").resolve()
    preview_json = (
        args.preview_json or root / "data" / "splits" / "fruit" / "dry_run_preview.json"
    ).resolve()

    class_names = load_class_names(fruit_raw)
    groups = collect_groups(fruit_raw, class_names)
    split_assignments = stratified_group_split(
        groups,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    summary = summarize_split(groups, split_assignments, class_names)

    print_summary(summary)

    preview_payload = {
        **summary,
        "seed": args.seed,
        "ratios": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
        },
        "grouping_key": "filename stem before .rf.",
        "mode": "dry-run" if args.dry_run else "write",
    }
    preview_json.parent.mkdir(parents=True, exist_ok=True)
    preview_json.write_text(json.dumps(preview_payload, indent=2), encoding="utf-8")
    print()
    print(f"Preview JSON saved to: {preview_json}")

    if args.write:
        write_split_outputs(
            groups,
            split_assignments,
            output_dir,
            class_names,
            summary,
            seed=args.seed,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
        )
        print(f"YOLO split written to: {output_dir}")
    else:
        print("Dry run only — no files copied. Re-run with --write after review.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
