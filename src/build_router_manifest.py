"""Build router train/val/test manifests from existing leaf and fruit splits.

Combines leaf CSVs (step 3) with fruit image folders (step 2) 1:1 per split.
No re-pooling or re-splitting.

Usage:
    python src/build_router_manifest.py
    python src/build_router_manifest.py --write
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
DOMAIN_TO_IDX = {"leaf": 0, "fruit": 1}
IDX_TO_DOMAIN = {0: "leaf", 1: "fruit"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_image_path(image_path: Path, root: Path) -> str:
    resolved = image_path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def collect_fruit_images(fruit_split_dir: Path) -> list[Path]:
    if not fruit_split_dir.is_dir():
        raise FileNotFoundError(f"Missing fruit split directory: {fruit_split_dir}")
    return sorted(
        path
        for path in fruit_split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_split_rows(root: Path, split: str) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []

    leaf_manifest = root / "data" / "splits" / "leaf" / f"{split}.csv"
    if not leaf_manifest.exists():
        raise FileNotFoundError(f"Missing leaf manifest: {leaf_manifest}")

    with leaf_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = root / row["image_path"]
            if not image_path.exists():
                raise FileNotFoundError(f"Leaf image not found: {image_path}")
            rows.append((relative_image_path(image_path, root), "leaf", 0))

    fruit_split_dir = root / "data" / "splits" / "fruit" / "images" / split
    for image_path in collect_fruit_images(fruit_split_dir):
        rows.append((relative_image_path(image_path, root), "fruit", 1))

    return rows


def summarize(rows_by_split: dict[str, list[tuple[str, str, int]]]) -> dict:
    summary = {"splits": {}, "totals": {"images": 0}}
    for split_name in SPLIT_NAMES:
        rows = rows_by_split[split_name]
        domain_counts = Counter(domain for _, domain, _ in rows)
        summary["splits"][split_name] = {
            "images": len(rows),
            "leaf": domain_counts.get("leaf", 0),
            "fruit": domain_counts.get("fruit", 0),
        }
        summary["totals"]["images"] += len(rows)
    return summary


def print_summary(summary: dict) -> None:
    print("=== Router manifest preview ===")
    print(f"{'Split':<6} {'Total':>7} {'Leaf':>7} {'Fruit':>7}")
    for split_name in SPLIT_NAMES:
        split = summary["splits"][split_name]
        print(
            f"{split_name:<6} {split['images']:>7} {split['leaf']:>7} {split['fruit']:>7}"
        )
    print()
    print(f"Grand total images: {summary['totals']['images']}")


def write_manifests(
    rows_by_split: dict[str, list[tuple[str, str, int]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "domain_map.json").write_text(
        json.dumps(
            {
                "domain_to_idx": DOMAIN_TO_IDX,
                "idx_to_domain": IDX_TO_DOMAIN,
                "class_names": ["leaf", "fruit"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for split_name in SPLIT_NAMES:
        manifest_path = output_dir / f"{split_name}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "domain", "domain_idx"])
            for image_path, domain, domain_idx in sorted(
                rows_by_split[split_name], key=lambda row: (row[1], row[0])
            ):
                writer.writerow([image_path, domain, domain_idx])

    summary = summarize(rows_by_split)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build router CSV manifests.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write manifests to data/splits/router/ (default: preview only).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = repo_root()
    output_dir = (args.output_dir or root / "data" / "splits" / "router").resolve()

    rows_by_split = {split: build_split_rows(root, split) for split in SPLIT_NAMES}
    summary = summarize(rows_by_split)
    print_summary(summary)

    if args.write:
        write_manifests(rows_by_split, output_dir)
        print(f"Router manifests written to: {output_dir}")
    else:
        print("Preview only — re-run with --write to save CSVs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
