"""Contract tests for the four-class classifier output (this branch's JSON schema)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_IMAGES = ROOT / "data/splits/fruit/images/test"
README_FRUIT_IMAGE = TEST_IMAGES / "Blossom_end_root_15-Copy_zoom_4_png.rf.8bd26be077f49081bbbe121a5e00408f.jpg"
ANTHRACNOSE_IMAGE = TEST_IMAGES / "anthracnose13-Copy_jpg.rf.61c4459e45351a47d0066563c6d86461.jpg"
FOUR = ["Early blight", "Blossom_End_Rot", "Healthy leaf", "Healthy plant"]

# Original keys and types of FourClassDiagnosticClassifier.classify(): the contract.
SCHEMA_V1 = {
    "predicted_class": (str,),
    "confidence": (float, int),
    "domain": (str,),
    "router_confidence": (float, int),
    "class_scores": (dict,),
    "detections": (list, type(None)),
    "raw_leaf_top3": (list, type(None)),
    "model_versions": (dict,),
}


def assert_v1_schema(result: dict) -> None:
    for key, types in SCHEMA_V1.items():
        assert key in result, f"missing original key {key!r}"
        assert isinstance(result[key], types), key
    assert result["predicted_class"] in FOUR
    assert list(result["class_scores"]) == FOUR
    assert set(result["model_versions"]) == {"router", "leaf", "fruit"}
    json.dumps(result)


@pytest.fixture(scope="module")
def clf():
    from four_class_infer import FourClassDiagnosticClassifier

    return FourClassDiagnosticClassifier(device="cpu")


def test_readme_blossom_end_rot(clf):
    result = clf.classify(README_FRUIT_IMAGE)
    assert_v1_schema(result)
    assert result["predicted_class"] == "Blossom_End_Rot"
    assert result["confidence"] == pytest.approx(0.693, abs=0.005)
    assert result["model_versions"]["fruit"] == "fruit_v1_yolov8n"
    assert result["warnings"] == []
    assert result["advice"]["condition"] == "Blossom-end rot"


def test_non_target_condition_is_flagged(clf):
    result = clf.classify(ANTHRACNOSE_IMAGE)
    assert_v1_schema(result)
    # Backward-compatible value...
    assert result["predicted_class"] == "Healthy plant"
    # ...but the additive keys say what was really seen.
    assert "other_condition_detected" in result["warnings"]
    assert result["other_conditions"][0]["class"] == "Anthracnose"
    assert result["advice"]["condition"].startswith("Anthracnose")


def test_no_detection_is_flagged(clf):
    result = clf.classify_image(Image.new("RGB", (640, 640), (200, 60, 40)))
    assert_v1_schema(result)
    if result["domain"] == "fruit" and not result["detections"]:
        assert "no_fruit_detected" in result["warnings"]
        assert result["advice"]["condition"] == "No fruit detected"


def test_leaf_branch_reports_other_leaf_condition(clf):
    tensor = clf._transform(Image.open(README_FRUIT_IMAGE).convert("RGB")).unsqueeze(0)
    predicted, confidence, scores, top3 = clf._classify_leaf(tensor)
    assert predicted in ("Early blight", "Healthy leaf")
    assert len(top3) == 3 and top3[0]["score"] >= top3[1]["score"]


def test_missing_checkpoint_raises():
    from four_class_infer import FourClassDiagnosticClassifier

    with pytest.raises(FileNotFoundError):
        FourClassDiagnosticClassifier(device="cpu", leaf_checkpoint="checkpoints/missing.pt")
