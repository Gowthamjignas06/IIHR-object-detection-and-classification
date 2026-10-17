"""End-to-end inference: router -> leaf | fruit -> severity -> unified JSON schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from advice import build_advice, load_knowledge_base
from evaluate import load_leaf_checkpoint, load_router_checkpoint, repo_root
from gradcam import compute_gradcam, save_gradcam_overlay
from severity import FruitDetection, fruit_severity_pct, leaf_severity_pct
from transforms import build_leaf_transforms

DOMAIN_NAMES = ("leaf", "fruit")
LEAF_MODEL_VERSION = "leaf_v1_efficientnet_b0"
FRUIT_MODEL_VERSION = "fruit_v1_yolov8n"
ROUTER_MODEL_VERSION = "router_v1_mobilenet_v3_small"

# model_version in the JSON output is the checkpoint filename stem, so a
# v2 checkpoint (e.g. leaf_v2_efficientnet_b0.pt) reports itself as v2.
DEFAULT_CHECKPOINTS = {
    "router": Path("checkpoints") / f"{ROUTER_MODEL_VERSION}.pt",
    "leaf": Path("checkpoints") / f"{LEAF_MODEL_VERSION}.pt",
    "fruit": Path("checkpoints") / f"{FRUIT_MODEL_VERSION}.pt",
}


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_image(path: Path) -> Image.Image:
    """Open an image upright (EXIF orientation applied) as RGB."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class TomatoInferencePipeline:
    def __init__(
        self,
        root: Path | None = None,
        device: str = "auto",
        gradcam_dir: Path | None = None,
        router_checkpoint: Path | str | None = None,
        leaf_checkpoint: Path | str | None = None,
        fruit_checkpoint: Path | str | None = None,
        with_advice: bool = True,
        knowledge_base_path: Path | str | None = None,
    ) -> None:
        self.root = root or repo_root()
        self.device = resolve_device(device)
        self.gradcam_dir = gradcam_dir or (self.root / "results" / "gradcam_overlays")
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)

        router_ckpt = self._resolve_checkpoint("router", router_checkpoint)
        leaf_ckpt = self._resolve_checkpoint("leaf", leaf_checkpoint)
        fruit_ckpt = self._resolve_checkpoint("fruit", fruit_checkpoint)
        self.router_model_version = router_ckpt.stem
        self.leaf_model_version = leaf_ckpt.stem
        self.fruit_model_version = fruit_ckpt.stem

        self.router, _ = load_router_checkpoint(router_ckpt, self.device)
        self.router.eval()

        self.leaf_model, self.leaf_checkpoint = load_leaf_checkpoint(leaf_ckpt, self.device)
        self.leaf_model.eval()
        self.leaf_class_names: list[str] = self.leaf_checkpoint["class_names"]
        self.leaf_transform = build_leaf_transforms(train=False, image_size=224)

        self.fruit_model = YOLO(str(fruit_ckpt))
        fruit_names = self.fruit_model.names
        self.fruit_class_names: list[str] = [fruit_names[i] for i in sorted(fruit_names)]

        # Explanation / treatment / action-step guidance (additive "advice" key).
        self.knowledge_base = load_knowledge_base(knowledge_base_path) if with_advice else None

    def _resolve_checkpoint(self, name: str, override: Path | str | None) -> Path:
        path = Path(override) if override is not None else DEFAULT_CHECKPOINTS[name]
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {path}. "
                "See checkpoints/README.md or pass the path explicitly."
            )
        return path.resolve()

    def _resolve_image_path(self, image_path: Path | str) -> Path:
        """Absolute paths as-is; relative paths try the CWD first, then the repo root."""
        path = Path(image_path)
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path.resolve()
        return (self.root / path).resolve()

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
            "model_version": self.leaf_model_version,
        }

    def _infer_fruit(self, image: Image.Image) -> dict:
        # Pass the already-loaded (EXIF-corrected) image so YOLO sees the same
        # pixels as the router.
        results = self.fruit_model.predict(
            image,
            conf=0.25,
            iou=0.7,
            device=str(self.device),
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
            "model_version": self.fruit_model_version,
        }

    def infer(self, image_path: Path | str) -> dict:
        path = self._resolve_image_path(image_path)
        return self.infer_image(load_image(path), path)

    def infer_image(self, image: Image.Image, image_path: Path | None = None) -> dict:
        """Run the pipeline on an in-memory RGB image (``image_path`` names the Grad-CAM file)."""
        image = image.convert("RGB")
        domain, route_conf = self._route(image)
        if domain == "leaf":
            result = self._infer_leaf(image, image_path)
        else:
            result = self._infer_fruit(image)

        # Additive keys only — the seven original keys keep their values.
        warnings: list[str] = []
        if domain == "fruit" and not result["detections"]:
            # "class" stays "Healthy_Tomato" (conf 0.0) for backward compatibility,
            # but nothing was actually detected.
            warnings.append("no_fruit_detected")
        result["router_confidence"] = round(route_conf, 4)
        result["warnings"] = warnings
        if self.knowledge_base is not None:
            result["advice"] = build_advice(result, self.knowledge_base)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tomato disease inference pipeline.")
    parser.add_argument("images", nargs="+", type=Path, help="Image paths to infer.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--router-checkpoint", type=Path, default=None)
    parser.add_argument("--leaf-checkpoint", type=Path, default=None)
    parser.add_argument("--fruit-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--no-advice",
        action="store_true",
        help="Omit the disease explanation / treatment / action-step 'advice' key.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pipeline = TomatoInferencePipeline(
        device=args.device,
        router_checkpoint=args.router_checkpoint,
        leaf_checkpoint=args.leaf_checkpoint,
        fruit_checkpoint=args.fruit_checkpoint,
        with_advice=not args.no_advice,
    )
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
