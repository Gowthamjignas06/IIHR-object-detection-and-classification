"""PyTorch Dataset classes for the leaf branch."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from transforms import build_leaf_transforms


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_class_map(splits_leaf_dir: Path) -> tuple[list[str], dict[str, int]]:
    class_map_path = splits_leaf_dir / "class_map.json"
    if not class_map_path.exists():
        raise FileNotFoundError(
            f"Missing {class_map_path}. Run src/split_leaf_manifest.py --write first."
        )
    payload = json.loads(class_map_path.read_text(encoding="utf-8"))
    class_names: list[str] = payload["class_names"]
    class_to_idx: dict[str, int] = payload["class_to_idx"]
    return class_names, class_to_idx


class LeafDataset(Dataset):
    """Image classification dataset backed by a CSV manifest (image_path, label)."""

    def __init__(
        self,
        manifest_path: Path | str,
        transform=None,
        root: Path | str | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else repo_root()
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []

        if class_to_idx is None:
            _, class_to_idx = load_class_map(self.root / "data" / "splits" / "leaf")
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}

        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != {"image_path", "label"}:
                raise ValueError(
                    f"{self.manifest_path} must have columns: image_path, label"
                )
            for row in reader:
                image_path = self.root / row["image_path"]
                label_name = row["label"]
                if label_name not in self.class_to_idx:
                    raise ValueError(f"Unknown label {label_name!r} in {self.manifest_path}")
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                self.samples.append(
                    (image_path, self.class_to_idx[label_name], label_name)
                )

        if not self.samples:
            raise ValueError(f"No samples loaded from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label_idx, _ = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx

    @property
    def class_names(self) -> list[str]:
        return [self.idx_to_class[idx] for idx in sorted(self.idx_to_class)]


def compute_class_weights(root: Path | str | None = None) -> torch.Tensor:
    """Inverse-frequency class weights from the train manifest (mean weight = 1)."""
    root_path = Path(root) if root is not None else repo_root()
    splits_dir = root_path / "data" / "splits" / "leaf"
    class_names, _ = load_class_map(splits_dir)
    counts = manifest_class_distribution(splits_dir / "train.csv")
    total = sum(counts[name] for name in class_names)
    num_classes = len(class_names)
    weights = [total / (num_classes * counts[name]) for name in class_names]
    return torch.tensor(weights, dtype=torch.float32)


def build_leaf_dataloader(
    split: str,
    *,
    batch_size: int = 16,
    shuffle: bool | None = None,
    num_workers: int = 0,
    image_size: int = 224,
    root: Path | str | None = None,
) -> DataLoader:
    root_path = Path(root) if root is not None else repo_root()
    splits_dir = root_path / "data" / "splits" / "leaf"
    manifest_path = splits_dir / f"{split}.csv"
    class_names, class_to_idx = load_class_map(splits_dir)
    train_mode = split == "train"
    if shuffle is None:
        shuffle = train_mode

    dataset = LeafDataset(
        manifest_path=manifest_path,
        transform=build_leaf_transforms(train=train_mode, image_size=image_size),
        root=root_path,
        class_to_idx=class_to_idx,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def manifest_class_distribution(manifest_path: Path | str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counts[row["label"]] += 1
    return counts


def run_sanity_check(batch_size: int = 8) -> None:
    root = repo_root()
    splits_dir = root / "data" / "splits" / "leaf"
    class_names, _ = load_class_map(splits_dir)

    print("=== Leaf dataset sanity check ===")
    print(f"Classes ({len(class_names)}): {class_names}")
    print()

    for split in ("train", "val", "test"):
        manifest = splits_dir / f"{split}.csv"
        dist = manifest_class_distribution(manifest)
        print(f"--- {split} manifest ({sum(dist.values())} images) ---")
        for label in class_names:
            print(f"  {label}: {dist.get(label, 0)}")
        print()

    loader = build_leaf_dataloader("train", batch_size=batch_size, num_workers=0)
    images, labels = next(iter(loader))
    print("Sample train batch:")
    print(f"  images.shape: {tuple(images.shape)} dtype={images.dtype}")
    print(f"  labels.shape: {tuple(labels.shape)} dtype={labels.dtype}")
    print(f"  label ids in batch: {labels.tolist()}")
    print(f"  pixel range: [{images.min().item():.3f}, {images.max().item():.3f}]")



def load_router_domain_map(splits_router_dir: Path) -> tuple[list[str], dict[str, int]]:
    domain_map_path = splits_router_dir / "domain_map.json"
    if not domain_map_path.exists():
        raise FileNotFoundError(
            f"Missing {domain_map_path}. Run src/build_router_manifest.py --write first."
        )
    payload = json.loads(domain_map_path.read_text(encoding="utf-8"))
    class_names: list[str] = payload["class_names"]
    domain_to_idx: dict[str, int] = payload["domain_to_idx"]
    return class_names, domain_to_idx


class RouterDataset(Dataset):
    """Binary leaf-vs-fruit dataset backed by router CSV manifests."""

    def __init__(
        self,
        manifest_path: Path | str,
        transform=None,
        root: Path | str | None = None,
        domain_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else repo_root()
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []

        if domain_to_idx is None:
            _, domain_to_idx = load_router_domain_map(
                self.root / "data" / "splits" / "router"
            )
        self.domain_to_idx = domain_to_idx
        self.idx_to_domain = {idx: name for name, idx in domain_to_idx.items()}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected = {"image_path", "domain", "domain_idx"}
            if reader.fieldnames is None or set(reader.fieldnames) != expected:
                raise ValueError(
                    f"{self.manifest_path} must have columns: image_path, domain, domain_idx"
                )
            for row in reader:
                image_path = self.root / row["image_path"]
                domain = row["domain"]
                if domain not in self.domain_to_idx:
                    raise ValueError(f"Unknown domain {domain!r} in {self.manifest_path}")
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                self.samples.append(
                    (image_path, self.domain_to_idx[domain], domain)
                )

        if not self.samples:
            raise ValueError(f"No samples loaded from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, domain_idx, _ = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, domain_idx

    @property
    def domain_names(self) -> list[str]:
        return [self.idx_to_domain[idx] for idx in sorted(self.idx_to_domain)]


def build_equal_domain_sampler(dataset: RouterDataset) -> WeightedRandomSampler:
    domain_counts = Counter(domain_idx for _, domain_idx, _ in dataset.samples)
    sample_weights = [
        1.0 / domain_counts[domain_idx] for _, domain_idx, _ in dataset.samples
    ]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_router_dataloader(
    split: str,
    *,
    batch_size: int = 16,
    shuffle: bool | None = None,
    equal_domain_sampling: bool = False,
    num_workers: int = 0,
    image_size: int = 224,
    root: Path | str | None = None,
) -> DataLoader:
    root_path = Path(root) if root is not None else repo_root()
    splits_dir = root_path / "data" / "splits" / "router"
    manifest_path = splits_dir / f"{split}.csv"
    _, domain_to_idx = load_router_domain_map(splits_dir)
    train_mode = split == "train"

    dataset = RouterDataset(
        manifest_path=manifest_path,
        transform=build_leaf_transforms(train=train_mode, image_size=image_size),
        root=root_path,
        domain_to_idx=domain_to_idx,
    )

    sampler = None
    if equal_domain_sampling and split == "train":
        sampler = build_equal_domain_sampler(dataset)
        shuffle = False
    elif shuffle is None:
        shuffle = train_mode

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    run_sanity_check()
