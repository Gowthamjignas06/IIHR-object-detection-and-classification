"""Disease explanation, treatment plan and action steps for a pipeline result.

Content lives in ``knowledge/disease_advice.yaml`` (one entry per model class).
``build_advice`` turns a unified-JSON result into the additive ``advice`` key.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_PATH = REPO_ROOT / "knowledge" / "disease_advice.yaml"
NO_DETECTION_KEY = "_no_detection"
URGENCY_ORDER = ("none", "low", "moderate", "high")
LOW_CONFIDENCE = 0.5


def load_knowledge_base(path: Path | str | None = None) -> dict:
    kb = yaml.safe_load(Path(path or DEFAULT_KB_PATH).read_text(encoding="utf-8"))
    if "conditions" not in kb:
        raise ValueError(f"{path or DEFAULT_KB_PATH} has no 'conditions' section")
    return kb


def _clamp_urgency(level: str, lowest: str, highest: str) -> str:
    rank = URGENCY_ORDER.index(level)
    rank = max(rank, URGENCY_ORDER.index(lowest))
    rank = min(rank, URGENCY_ORDER.index(highest))
    return URGENCY_ORDER[rank]


def urgency_for(entry: dict, severity_pct: float, tiers: dict) -> tuple[str, str]:
    """Urgency from severity, bounded by the condition's min/max urgency."""
    lowest = entry.get("min_urgency", "none")
    highest = entry.get("max_urgency", "high")
    if entry["category"] in ("healthy", "unknown"):
        return lowest, "No disease detected."

    if severity_pct < tiers.get("low", 5.0):
        level = "low"
    elif severity_pct < tiers.get("moderate", 25.0):
        level = "moderate"
    else:
        level = "high"
    clamped = _clamp_urgency(level, lowest, highest)

    reason = f"Estimated affected area {severity_pct:.1f}% ({level} by area)."
    if clamped != level:
        if URGENCY_ORDER.index(clamped) > URGENCY_ORDER.index(level):
            reason += f" Raised to {clamped}: {entry['category']} condition that spreads or cannot be cured."
        else:
            reason += f" Capped at {clamped}: {entry['category']} condition that does not spread between plants."
    return clamped, reason


def build_advice(result: dict, kb: dict) -> dict:
    """Build the ``advice`` dict for one pipeline result (additive JSON key)."""
    conditions = kb["conditions"]
    no_detection = "no_fruit_detected" in (result.get("warnings") or [])
    key = NO_DETECTION_KEY if no_detection else result["class"]
    entry = conditions.get(key)
    if entry is None:
        return {
            "available": False,
            "condition": result["class"],
            "notes": [f"No guidance in the knowledge base for class {result['class']!r}."],
            "get_help": kb.get("get_help", ""),
            "disclaimer": kb.get("disclaimer", ""),
            "knowledge_base_version": kb.get("version"),
        }

    urgency, urgency_reason = urgency_for(
        entry, float(result.get("severity_pct") or 0.0), kb.get("urgency_tiers", {})
    )

    notes: list[str] = []
    confidence = float(result.get("confidence") or 0.0)
    if not no_detection and confidence < LOW_CONFIDENCE:
        notes.append(
            f"Low model confidence ({confidence:.0%}): treat this diagnosis as uncertain "
            "and confirm it before acting."
        )
    also_detected: list[str] = []
    if result.get("domain") == "fruit" and result.get("detections"):
        seen = {result["class"]}
        for det in result["detections"]:
            name = det["class"]
            if name not in seen and name != "Healthy_Tomato":
                seen.add(name)
                label = conditions.get(name, {}).get("display_name", name)
                also_detected.append(f"{label} ({det['confidence']:.0%})")
    if also_detected:
        notes.append(
            "Also detected in this photo: " + ", ".join(also_detected)
            + ". Check those sections of the guide as well."
        )
    if not kb.get("reviewed", False):
        notes.append("This guidance has not yet been reviewed by an agronomist.")

    treatment = entry.get("treatment") or {}
    steps = entry.get("action_steps") or {}
    return {
        "available": True,
        "condition": entry["display_name"],
        "category": entry["category"],
        "contagious": bool(entry.get("contagious", False)),
        "explanation": entry["explanation"],
        "cause": entry["cause"],
        "symptoms": list(entry.get("symptoms") or []),
        "why_it_matters": entry["why_it_matters"],
        "urgency": urgency,
        "urgency_reason": urgency_reason,
        "treatment_plan": {
            "cultural": list(treatment.get("cultural") or []),
            "organic": list(treatment.get("organic") or []),
            "chemical": list(treatment.get("chemical") or []),
        },
        "action_steps": {
            "immediate": list(steps.get("immediate") or []),
            "this_week": list(steps.get("this_week") or []),
            "prevention": list(steps.get("prevention") or []),
        },
        "notes": notes,
        "get_help": kb.get("get_help", ""),
        "disclaimer": kb.get("disclaimer", ""),
        "knowledge_base_version": kb.get("version"),
    }


def advice_to_markdown(advice: dict) -> str:
    """Render the advice dict for the Gradio UI."""
    if not advice:
        return ""
    lines = [f"## Disease guide — {advice['condition']}"]
    if not advice.get("available", False):
        lines += [*(f"- {n}" for n in advice.get("notes", [])), "", f"_{advice.get('disclaimer', '')}_"]
        return "\n".join(lines)

    urgency_badge = {
        "none": "🟢 No action needed",
        "low": "🟡 Low urgency",
        "moderate": "🟠 Moderate urgency",
        "high": "🔴 High urgency",
    }.get(advice["urgency"], advice["urgency"])
    category = {
        "fungal": "fungal disease",
        "oomycete": "water-mould disease (oomycete)",
        "bacterial": "bacterial disease",
        "viral": "virus disease",
        "pest": "pest (mites)",
        "physiological": "physiological disorder, not an infection",
    }.get(advice["category"], advice["category"])
    spread = "spreads between plants" if advice["contagious"] else "does not spread between plants"
    lines += [
        f"**{urgency_badge}** — {advice['urgency_reason']}  ",
        f"Type: **{category}** ({spread})",
        "",
        advice["explanation"],
        "",
        f"**Cause:** {advice['cause']}",
    ]
    if advice["symptoms"]:
        lines += ["", "**What to look for**", *(f"- {s}" for s in advice["symptoms"])]
    lines += ["", f"**Why it matters:** {advice['why_it_matters']}"]

    plan = advice["treatment_plan"]
    if any(plan.values()):
        lines += ["", "### Treatment plan"]
        for key, title in (
            ("cultural", "Cultural / field practices"),
            ("organic", "Organic / biological options"),
            ("chemical", "Chemical options (registered products only)"),
        ):
            if plan[key]:
                lines += [f"**{title}**", *(f"- {item}" for item in plan[key]), ""]

    steps = advice["action_steps"]
    if any(steps.values()):
        lines += ["### Action steps"]
        for key, title in (
            ("immediate", "Now"),
            ("this_week", "This week"),
            ("prevention", "Next season / prevention"),
        ):
            if steps[key]:
                lines += [f"**{title}**", *(f"{i}. {s}" for i, s in enumerate(steps[key], 1)), ""]

    if advice["notes"]:
        lines += ["**Notes**", *(f"- {n}" for n in advice["notes"]), ""]
    lines += [f"**Get help:** {advice['get_help']}", "", f"_{advice['disclaimer']}_"]
    return "\n".join(lines)
