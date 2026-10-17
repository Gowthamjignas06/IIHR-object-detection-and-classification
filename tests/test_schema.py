"""Contract tests for the unified JSON output and the fruit severity heuristic.

Run from the repo root:  python -m pytest tests/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from severity import FruitDetection, _covered_area, fruit_severity_pct  # noqa: E402

README_FRUIT_IMAGE = (
    ROOT
    / "data/splits/fruit/images/test"
    / "Blossom_end_root_15-Copy_zoom_4_png.rf.8bd26be077f49081bbbe121a5e00408f.jpg"
)

# The original seven keys and their allowed types: the backward-compatible contract.
SCHEMA_V1 = {
    "domain": (str,),
    "class": (str,),
    "confidence": (float, int),
    "severity_pct": (float, int),
    "gradcam_overlay": (str, type(None)),
    "detections": (list, type(None)),
    "model_version": (str,),
}


def assert_v1_schema(result: dict) -> None:
    for key, types in SCHEMA_V1.items():
        assert key in result, f"missing original key {key!r}"
        assert isinstance(result[key], types), f"{key!r} has type {type(result[key]).__name__}"
    assert result["domain"] in ("leaf", "fruit")
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["severity_pct"] <= 100.0
    json.dumps(result)  # must stay JSON-serialisable


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from infer_pipeline import TomatoInferencePipeline

    return TomatoInferencePipeline(
        device="cpu", gradcam_dir=tmp_path_factory.mktemp("gradcam")
    )


def test_readme_fruit_example(pipeline):
    result = pipeline.infer(README_FRUIT_IMAGE)
    assert_v1_schema(result)
    assert result["domain"] == "fruit"
    assert result["class"] == "Blossom_End_Rot"
    assert result["confidence"] == pytest.approx(0.693, abs=0.005)
    assert result["gradcam_overlay"] is None
    assert result["model_version"] == "fruit_v1_yolov8n"
    for det in result["detections"]:
        assert set(det) == {"class", "confidence", "bbox"}
        assert len(det["bbox"]) == 4
    assert isinstance(result["router_confidence"], float)
    assert result["warnings"] == []
    advice = result["advice"]
    assert advice["available"] and advice["condition"] == "Blossom-end rot"
    assert advice["category"] == "physiological" and advice["urgency"] == "moderate"


def test_leaf_branch_schema(pipeline):
    # No PlantVillage images ship with the repo; any RGB image exercises the leaf
    # code path (classifier, Grad-CAM, severity).
    image = Image.open(README_FRUIT_IMAGE).convert("RGB")
    result = pipeline._infer_leaf(image, Path("leaf_branch_test.jpg"))
    assert_v1_schema(result)
    assert result["domain"] == "leaf"
    assert result["class"] in pipeline.leaf_class_names
    assert result["detections"] is None
    assert Path(result["gradcam_overlay"]).exists()
    assert result["model_version"] == "leaf_v1_efficientnet_b0"


def test_no_detection_is_flagged(pipeline):
    blank = Image.new("RGB", (640, 640), (200, 60, 40))
    result = pipeline.infer_image(blank)
    assert_v1_schema(result)
    if result["domain"] == "fruit" and not result["detections"]:
        # Backward-compatible values, plus the additive warning.
        assert result["class"] == "Healthy_Tomato"
        assert result["confidence"] == 0.0
        assert "no_fruit_detected" in result["warnings"]


def test_missing_checkpoint_raises():
    from infer_pipeline import TomatoInferencePipeline

    with pytest.raises(FileNotFoundError):
        TomatoInferencePipeline(device="cpu", fruit_checkpoint="checkpoints/does_not_exist.pt")


def test_covered_area_counts_overlap_once():
    assert _covered_area([(0, 0, 10, 10)]) == 100
    assert _covered_area([(0, 0, 10, 10), (20, 20, 30, 30)]) == 200
    assert _covered_area([(0, 0, 10, 10), (5, 0, 15, 10)]) == 150
    assert _covered_area([(0, 0, 10, 10), (2, 2, 4, 4)]) == 100


def test_fruit_severity_overlapping_boxes_not_capped():
    dets = [
        FruitDetection("Late_Blight", 0.9, (0, 0, 100, 100)),
        FruitDetection("Late_Blight", 0.8, (50, 0, 150, 100)),
        FruitDetection("Healthy_Tomato", 0.7, (0, 0, 60, 60)),
    ]
    # Disease union = 150x100; fruit area = union of all boxes = 150x100.
    assert fruit_severity_pct(dets, (640, 640)) == 100.0
    dets.append(FruitDetection("Healthy_Tomato", 0.7, (150, 0, 300, 100)))
    # Fruit area grows to 300x100 -> 50%.
    assert fruit_severity_pct(dets, (640, 640)) == 50.0


def test_fruit_severity_no_disease_is_zero():
    dets = [FruitDetection("Healthy_Tomato", 0.9, (0, 0, 100, 100))]
    assert fruit_severity_pct(dets, (640, 640)) == 0.0
    assert fruit_severity_pct([], (640, 640)) == 0.0
