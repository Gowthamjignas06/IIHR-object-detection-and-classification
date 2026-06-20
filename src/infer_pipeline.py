"""End-to-end inference: router -> leaf | fruit -> severity -> unified JSON schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from ultralytics import YOLO

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evaluate import load_leaf_checkpoint, load_router_checkpoint, repo_root
from gradcam import compute_gradcam, save_gradcam_overlay
from models.leaf_classifier import build_leaf_classifier
from models.router import build_router
from severity import FruitDetection, fruit_severity_pct, leaf_severity_pct
from transforms import build_leaf_transforms
from train_fruit_yolo import resolve_data_yaml, write_resolved_data_yaml

DOMAIN_NAMES = ("leaf", "fruit")
LEAF_MODEL_VERSION = "leaf_v1_efficientnet_b0"
FRUIT_MODEL_VERSION = "fruit_v1_yolov8n"
ROUTER_MODEL_VERSION = "router_v1_mobilenet_v3_small"


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class TomatoInferencePipeline:
    def __init__(
        self,
        root: Path | None = None,
        device: str = "auto",
        gradcam_dir: Path | None = None,
    ) -> None:
        self.root = root or repo_root()
        self.device = resolve_device(device)
        self.gradcam_dir = gradcam_dir or (self.root / "results" / "gradcam_overlays")
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)

        router_ckpt = self.root / "checkpoints" / f"{ROUTER_MODEL_VERSION}.pt"
        leaf_ckpt = self.root / "checkpoints" / f"{LEAF_MODEL_VERSION}.pt"
        fruit_ckpt = self.root / "checkpoints" / f"{FRUIT_MODEL_VERSION}.pt"
        if not fruit_ckpt.exists():
            fruit_ckpt = self.root / "runs" / "fruit" / "train" / "weights" / "best.pt"

        self.router, _ = load_router_checkpoint(router_ckpt, self.device)
        self.router.eval()

        self.leaf_model, self.leaf_checkpoint = load_leaf_checkpoint(leaf_ckpt, self.device)
        self.leaf_model.eval()
        self.leaf_class_names: list[str] = self.leaf_checkpoint["class_names"]
        self.leaf_transform = build_leaf_transforms(train=False, image_size=224)

        self.fruit_model = YOLO(str(fruit_ckpt))
        data_yaml = write_resolved_data_yaml(
            resolve_data_yaml(
                yaml.safe_load((self.root / "configs" / "fruit_yolo.yaml").read_text())
            )
        )
        fruit_names = yaml.safe_load(data_yaml.read_text())["names"]
        self.fruit_class_names: list[str] = list(fruit_names)

    def _route(self, image: Image.Image) -> tuple[str, float]:
        tensor = self.leaf_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.router(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
        return DOMAIN_NAMES[idx], float(probs[idx].item())

    def _infer_leaf(self, image: Image.Image, image_path: Path | None) -> dict:
        tensor = self.leaf_transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad_(False)
        with torch.no_grad():
            logits = self.leaf_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())

        cam_tensor = self.leaf_transform(image).unsqueeze(0).to(self.device)
        cam_tensor.requires_grad_(True)
        self.leaf_model.eval()
        cam = compute_gradcam(self.leaf_model, cam_tensor, pred_idx)
        stem = image_path.stem if image_path is not None else "sample"
        overlay_path = save_gradcam_overlay(
            image,
            cam,
            self.gradcam_dir / f"{stem}_gradcam.jpg",
        )
        severity = leaf_severity_pct(image)
        return {
            "domain": "leaf",
            "class": self.leaf_class_names[pred_idx],
            "confidence": round(confidence, 4),
            "severity_pct": severity,
            "gradcam_overlay": overlay_path,
            "detections": None,
            "model_version": LEAF_MODEL_VERSION,
        }

    def _infer_fruit(self, image: Image.Image, image_path: Path) -> dict:
        results = self.fruit_model.predict(
            str(image_path),
            conf=0.25,
            iou=0.7,
            verbose=False,
        )[0]
        detections: list[FruitDetection] = []
        if results.boxes is not None and len(results.boxes):
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_idx in zip(boxes, confs, classes):
                detections.append(
                    FruitDetection(
                        class_name=self.fruit_class_names[int(cls_idx)],
                        confidence=float(conf),
                        bbox_xyxy=tuple(float(v) for v in box),
                    )
                )

        if detections:
            primary = max(detections, key=lambda d: d.confidence)
            primary_class = primary.class_name
            primary_conf = primary.confidence
        else:
            primary_class = "Healthy_Tomato"
            primary_conf = 0.0

        severity = fruit_severity_pct(
            detections,
            (results.orig_shape[1], results.orig_shape[0]),
        )
        return {
            "domain": "fruit",
            "class": primary_class,
            "confidence": round(primary_conf, 4),
            "severity_pct": severity,
            "gradcam_overlay": None,
            "detections": [
                {
                    "class": det.class_name,
                    "confidence": round(det.confidence, 4),
                    "bbox": [round(v, 1) for v in det.bbox_xyxy],
                }
                for det in detections
            ],
            "model_version": FRUIT_MODEL_VERSION,
        }

    def infer(self, image_path: Path | str) -> dict:
        path = Path(image_path)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        image = Image.open(path).convert("RGB")
        domain, route_conf = self._route(image)
        if domain == "leaf":
            return self._infer_leaf(image, path)
        return self._infer_fruit(image, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tomato disease inference pipeline.")
    parser.add_argument("images", nargs="+", type=Path, help="Image paths to infer.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pipeline = TomatoInferencePipeline(device=args.device)
    outputs = []
    for image_path in args.images:
        result = pipeline.infer(image_path)
        outputs.append(result)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"=== {image_path} ===")
            print(json.dumps(result, indent=2))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
