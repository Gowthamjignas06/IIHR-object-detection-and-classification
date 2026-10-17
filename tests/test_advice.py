"""Knowledge-base coverage and advice-building tests (no models needed except coverage)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advice import (  # noqa: E402
    NO_DETECTION_KEY,
    URGENCY_ORDER,
    advice_to_markdown,
    build_advice,
    load_knowledge_base,
)

LEAF_CLASSES = json.loads((ROOT / "data/splits/leaf/class_map.json").read_text())["class_names"]
FRUIT_CLASSES = list(yaml.safe_load((ROOT / "data/splits/fruit/data.yaml").read_text())["names"])
CATEGORIES = {"fungal", "oomycete", "bacterial", "viral", "pest", "physiological", "healthy", "unknown"}


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


def _result(cls: str, domain: str, severity: float = 10.0, confidence: float = 0.9, **extra) -> dict:
    return {
        "domain": domain,
        "class": cls,
        "confidence": confidence,
        "severity_pct": severity,
        "gradcam_overlay": None,
        "detections": [] if domain == "fruit" else None,
        "model_version": "test",
        "warnings": [],
        **extra,
    }


def test_every_model_class_has_an_entry(kb):
    missing = [c for c in LEAF_CLASSES + FRUIT_CLASSES + [NO_DETECTION_KEY] if c not in kb["conditions"]]
    assert not missing, f"knowledge base missing: {missing}"


@pytest.mark.parametrize("cls", LEAF_CLASSES + FRUIT_CLASSES)
def test_entries_are_complete(kb, cls):
    entry = kb["conditions"][cls]
    for field in ("display_name", "category", "explanation", "cause", "why_it_matters"):
        assert entry.get(field), f"{cls}: empty {field}"
    assert entry["category"] in CATEGORIES
    assert entry["min_urgency"] in URGENCY_ORDER and entry["max_urgency"] in URGENCY_ORDER
    assert URGENCY_ORDER.index(entry["min_urgency"]) <= URGENCY_ORDER.index(entry["max_urgency"])
    assert entry["action_steps"]["immediate"], f"{cls}: no immediate action step"
    if entry["category"] != "healthy":
        assert entry["symptoms"], f"{cls}: no symptoms"
        assert any(entry["treatment"].values()), f"{cls}: empty treatment plan"
        assert entry["action_steps"]["prevention"], f"{cls}: no prevention steps"


@pytest.mark.parametrize("cls,domain", [(c, "leaf") for c in LEAF_CLASSES] + [(c, "fruit") for c in FRUIT_CLASSES])
def test_build_advice_for_every_class(kb, cls, domain):
    advice = build_advice(_result(cls, domain), kb)
    assert advice["available"] is True
    assert advice["urgency"] in URGENCY_ORDER
    json.dumps(advice)
    assert advice["condition"] in advice_to_markdown(advice)


def test_urgency_rules(kb):
    # Healthy: never urgent.
    assert build_advice(_result("Healthy", "leaf", severity=80), kb)["urgency"] == "none"
    # Viruses / late blight: always high, even with a tiny affected area.
    assert build_advice(_result("Tomato mosaic virus", "leaf", severity=1), kb)["urgency"] == "high"
    assert build_advice(_result("Late_Blight", "fruit", severity=1), kb)["urgency"] == "high"
    # Physiological disorders are capped (they do not spread).
    assert build_advice(_result("Fruit_Cracking", "fruit", severity=90), kb)["urgency"] == "moderate"
    # Fungal leaf disease scales with severity.
    levels = [build_advice(_result("Early blight", "leaf", severity=s), kb)["urgency"] for s in (2, 10, 40)]
    assert levels == ["low", "moderate", "high"]


def test_no_detection_and_low_confidence_notes(kb):
    advice = build_advice(_result("Healthy_Tomato", "fruit", confidence=0.0, warnings=["no_fruit_detected"]), kb)
    assert advice["condition"] == "No fruit detected"
    assert "Retake" in advice["action_steps"]["immediate"][0]

    advice = build_advice(_result("Early blight", "leaf", confidence=0.3), kb)
    assert any("Low model confidence" in n for n in advice["notes"])


def test_secondary_detections_are_mentioned(kb):
    result = _result(
        "Blossom_End_Rot",
        "fruit",
        detections=[
            {"class": "Blossom_End_Rot", "confidence": 0.69, "bbox": [0, 0, 1, 1]},
            {"class": "Catfaced", "confidence": 0.26, "bbox": [0, 0, 1, 1]},
            {"class": "Healthy_Tomato", "confidence": 0.5, "bbox": [0, 0, 1, 1]},
        ],
    )
    notes = " ".join(build_advice(result, kb)["notes"])
    assert "Catfacing" in notes and "Healthy" not in notes


def test_unknown_class_degrades_gracefully(kb):
    advice = build_advice(_result("Not a class", "leaf"), kb)
    assert advice["available"] is False
    assert advice_to_markdown(advice)
