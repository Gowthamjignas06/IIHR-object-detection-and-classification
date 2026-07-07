"""
Standalone four-class tomato diagnostic classifier.

Distinguishes exactly four conditions without any new training by routing
through the existing trained leaf and fruit models and extracting only the
four relevant scores.

Target classes (output labels):
  "Early blight"     — leaf fungal disease   (leaf model index 1)
  "Blossom_End_Rot"  — fruit abiotic disease  (YOLO class 1)
  "Healthy leaf"     — healthy tomato leaf    (leaf model index 2)
  "Healthy plant"    — healthy tomato plant   (YOLO class 4)

Usage:
  # Single image
  python src/four_class_infer.py path/to/image.jpg

  # Multiple images with JSON output
  python src/four_class_infer.py img1.jpg img2.jpg --json

  # Python API
  from src.four_class_infer import FourClassDiagnosticClassifier
  clf = FourClassDiagnosticClassifier()
  result = clf.classify("path/to/image.jpg")
"""

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
from train_fruit_yolo import resolve_data_yaml, write_resolved_data_yaml
from transforms import build_leaf_transforms

# ── Canonical four-class output labels ────────────────────────────────────────
FOUR_CLASSES: list[str] = [
    "Early blight",
    "Blossom_End_Rot",
    "Healthy leaf",
    "Healthy plant",
]

# ── Source-model indices for each target class ─────────────────────────────────
# Leaf EfficientNet-B0: index within 10-class softmax (from class_map.json)
_LEAF_EARLY_BLIGHT_IDX: int = 1   # "Early blight"
_LEAF_HEALTHY_IDX: int = 2        # "Healthy"

# Fruit YOLOv8n: YOLO class ID (from data/splits/fruit/data.yaml)
_FRUIT_BER_CLS: int = 1           # "Blossom_End_Rot"
_FRUIT_HEALTHY_CLS: int = 4       # "Healthy_Tomato"

# ── Model version strings ──────────────────────────────────────────────────────
_ROUTER_VERSION = "router_v1_mobilenet_v3_small"
_LEAF_VERSION = "leaf_v1_efficientnet_b0"
_FRUIT_VERSION = "fruit_v1_yolov8n"

_DOMAIN_NAMES = ("leaf", "fruit")


def _resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class FourClassDiagnosticClassifier:
    """
    Zero-retraining four-class wrapper around the existing trained pipeline.

    Routing logic:
      leaf  → EfficientNet-B0 softmax → early blight prob vs healthy-leaf prob
      fruit → YOLOv8n detections → Blossom_End_Rot conf vs Healthy_Tomato conf

    No weights are modified or fine-tuned; the three existing checkpoint files
    are loaded as-is.
    """

    def __init__(
        self,
        root: Path | None = None,
        device: str = "auto",
        fruit_conf_threshold: float = 0.20,
        fruit_iou_threshold: float = 0.70,
    ) -> None:
        self.root = root or repo_root()
        self.device = _resolve_device(device)
        self.fruit_conf_threshold = fruit_conf_threshold
        self.fruit_iou_threshold = fruit_iou_threshold

        router_ckpt = self.root / "checkpoints" / f"{_ROUTER_VERSION}.pt"
        leaf_ckpt = self.root / "checkpoints" / f"{_LEAF_VERSION}.pt"
        fruit_ckpt = self.root / "checkpoints" / f"{_FRUIT_VERSION}.pt"
        if not fruit_ckpt.exists():
            fruit_ckpt = self.root / "runs" / "fruit" / "train" / "weights" / "best.pt"

        for ckpt, label in [
            (router_ckpt, "router"),
            (leaf_ckpt, "leaf"),
            (fruit_ckpt, "fruit"),
        ]:
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"{label} checkpoint not found at {ckpt}. "
                    "Train the model first (see README) or download from the release."
                )

        self._router, _ = load_router_checkpoint(router_ckpt, self.device)
        self._router.eval()

        self._leaf_model, _leaf_data = load_leaf_checkpoint(leaf_ckpt, self.device)
        self._leaf_model.eval()
        self._leaf_class_names: list[str] = _leaf_data["class_names"]

        self._fruit_model = YOLO(str(fruit_ckpt))
        _data_yaml = write_resolved_data_yaml(
            resolve_data_yaml(
                yaml.safe_load(
                    (self.root / "configs" / "fruit_yolo.yaml").read_text()
                )
            )
        )
        self._fruit_class_names: list[str] = list(
            yaml.safe_load(_data_yaml.read_text())["names"]
        )

        self._transform = build_leaf_transforms(train=False, image_size=224)

    @property
    def class_names(self) -> list[str]:
        return FOUR_CLASSES.copy()

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route(self, tensor: torch.Tensor) -> tuple[str, float]:
        with torch.no_grad():
            probs = torch.softmax(self._router(tensor), dim=1)[0]
            idx = int(probs.argmax().item())
        return _DOMAIN_NAMES[idx], float(probs[idx].item())

    # ── Leaf branch ───────────────────────────────────────────────────────────

    def _classify_leaf(
        self, tensor: torch.Tensor
    ) -> tuple[str, float, dict[str, float], list[dict] | None]:
        """
        Run the 10-class leaf EfficientNet and extract the two target scores.

        The full softmax is computed; only "Early blight" and "Healthy" indices
        are surfaced in the four-class output. All other leaf probabilities are
        available in raw_leaf_probs for debugging.
        """
        with torch.no_grad():
            probs_all = torch.softmax(self._leaf_model(tensor), dim=1)[0]

        p_eb = float(probs_all[_LEAF_EARLY_BLIGHT_IDX].item())
        p_hl = float(probs_all[_LEAF_HEALTHY_IDX].item())

        predicted = "Early blight" if p_eb >= p_hl else "Healthy leaf"
        confidence = p_eb if predicted == "Early blight" else p_hl

        # Top-3 full leaf predictions for transparency
        top3_indices = probs_all.topk(3).indices.tolist()
        raw_leaf_top3 = [
            {
                "class": self._leaf_class_names[i],
                "score": round(float(probs_all[i].item()), 4),
            }
            for i in top3_indices
        ]

        scores: dict[str, float] = {
            "Early blight": round(p_eb, 4),
            "Blossom_End_Rot": 0.0,
            "Healthy leaf": round(p_hl, 4),
            "Healthy plant": 0.0,
        }
        return predicted, confidence, scores, raw_leaf_top3

    # ── Fruit branch ──────────────────────────────────────────────────────────

    def _classify_fruit(
        self, image_path: Path
    ) -> tuple[str, float, dict[str, float], list[dict]]:
        """
        Run YOLOv8n detection and surface Blossom_End_Rot and Healthy_Tomato
        confidence scores as the two fruit-side four-class scores.

        If neither target class is detected, defaults to Healthy plant (no visible
        pathology ≈ healthy).
        """
        results = self._fruit_model.predict(
            str(image_path),
            conf=self.fruit_conf_threshold,
            iou=self.fruit_iou_threshold,
            verbose=False,
        )[0]

        ber_conf: float = 0.0
        healthy_conf: float = 0.0
        detections: list[dict] = []

        if results.boxes is not None and len(results.boxes):
            for box_xyxy, conf, cls_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int),
            ):
                cls_name = self._fruit_class_names[int(cls_id)]
                detections.append(
                    {
                        "class": cls_name,
                        "confidence": round(float(conf), 4),
                        "bbox": [round(float(v), 1) for v in box_xyxy],
                    }
                )
                if int(cls_id) == _FRUIT_BER_CLS:
                    ber_conf = max(ber_conf, float(conf))
                elif int(cls_id) == _FRUIT_HEALTHY_CLS:
                    healthy_conf = max(healthy_conf, float(conf))

        # No target-class detection → treat as healthy
        if ber_conf == 0.0 and healthy_conf == 0.0:
            healthy_conf = 1.0

        predicted = "Blossom_End_Rot" if ber_conf >= healthy_conf else "Healthy plant"
        confidence = ber_conf if predicted == "Blossom_End_Rot" else healthy_conf

        scores: dict[str, float] = {
            "Early blight": 0.0,
            "Blossom_End_Rot": round(ber_conf, 4),
            "Healthy leaf": 0.0,
            "Healthy plant": round(healthy_conf, 4),
        }
        return predicted, confidence, scores, detections

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, image_path: Path | str) -> dict:
        """
        Classify an image into one of the four target conditions.

        Returns
        -------
        dict with keys:
          predicted_class  : str — one of FOUR_CLASSES
          confidence       : float 0-1
          domain           : "leaf" or "fruit"
          class_scores     : dict mapping each of the 4 class names to its score
          detections       : list of YOLO boxes (fruit branch only), else None
          raw_leaf_top3    : top-3 full leaf predictions (leaf branch only), else None
          model_versions   : dict of model version strings used
        """
        path = Path(image_path)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        image = Image.open(path).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self.device)

        domain, route_conf = self._route(tensor)

        if domain == "leaf":
            predicted, confidence, scores, raw_leaf_top3 = self._classify_leaf(tensor)
            detections = None
        else:
            predicted, confidence, scores, detections = self._classify_fruit(path)
            raw_leaf_top3 = None

        return {
            "predicted_class": predicted,
            "confidence": round(confidence, 4),
            "domain": domain,
            "router_confidence": round(route_conf, 4),
            "class_scores": scores,
            "detections": detections,
            "raw_leaf_top3": raw_leaf_top3,
            "model_versions": {
                "router": _ROUTER_VERSION,
                "leaf": _LEAF_VERSION,
                "fruit": _FRUIT_VERSION,
            },
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four-class tomato diagnostic classifier (no retraining).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "Target classes:",
                "  Early blight    — leaf fungal disease",
                "  Blossom_End_Rot — fruit abiotic disease",
                "  Healthy leaf    — healthy tomato leaf",
                "  Healthy plant   — healthy tomato plant",
            ]
        ),
    )
    parser.add_argument("images", nargs="+", type=Path, help="Image paths to classify.")
    parser.add_argument("--device", default="auto", help="Compute device (auto/cpu/cuda).")
    parser.add_argument(
        "--json", action="store_true", help="Print compact JSON output only."
    )
    parser.add_argument(
        "--fruit-conf", type=float, default=0.20,
        help="YOLO detection confidence threshold (default 0.20).",
    )
    return parser


def _pretty_print(result: dict, image_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Image : {image_path}")
    print(f"Domain: {result['domain']}  (router confidence: {result['router_confidence']:.4f})")
    print(f"\n  Predicted class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']:.4f}")
    print("\n  Four-class scores:")
    for cls, score in result["class_scores"].items():
        marker = " <-- predicted" if cls == result["predicted_class"] else ""
        print(f"    {cls:<20}  {score:.4f}{marker}")
    if result["raw_leaf_top3"]:
        print("\n  Full leaf model top-3:")
        for entry in result["raw_leaf_top3"]:
            print(f"    {entry['class']:<35}  {entry['score']:.4f}")
    if result["detections"]:
        print(f"\n  YOLO detections ({len(result['detections'])} box(es)):")
        for det in result["detections"]:
            print(
                f"    {det['class']:<25}  conf={det['confidence']:.4f}  bbox={det['bbox']}"
            )
    print()


def main() -> int:
    args = _build_parser().parse_args()
    clf = FourClassDiagnosticClassifier(
        device=args.device,
        fruit_conf_threshold=args.fruit_conf,
    )
    for img_path in args.images:
        result = clf.classify(img_path)
        if args.json:
            print(json.dumps(result))
        else:
            _pretty_print(result, img_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
