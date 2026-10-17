"""Run the pipeline on the external validation photos (inference only).

The Google Drive set in data/external_validation/ is for VALIDATION ONLY:
never train, fine-tune, or choose thresholds on it. This script only runs
inference and summarises what the pipeline did. The photos have no disease
labels, so it reports routing, detections, predicted classes and confidence
rather than accuracy.

Usage:
    python scripts/eval_external_validation.py
    python scripts/eval_external_validation.py --save-overlays
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infer_pipeline import TomatoInferencePipeline, load_image  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# What a correct pipeline should do with each Drive folder.
FOLDER_EXPECTATION = {
    "fruit pics": "fruit",
    "Article related_other pics": "out_of_scope",  # field/group/landscape photos
}


def summarise(rows: list[dict], expected: str) -> dict:
    n = len(rows)
    domains = Counter(r["domain"] for r in rows)
    summary = {
        "expected": expected,
        "n_images": n,
        "routed": dict(domains),
        "router_confidence_median": round(statistics.median(r["router_confidence"] for r in rows), 4) if rows else None,
        "predicted_class": dict(Counter(r["class_display"] for r in rows).most_common()),
        "confidence_median": round(statistics.median(r["confidence"] for r in rows), 4) if rows else None,
        "low_confidence_lt_0_5": sum(r["confidence"] < 0.5 for r in rows),
    }
    if expected == "fruit":
        summary["routed_to_fruit_pct"] = round(100 * domains.get("fruit", 0) / n, 1) if n else None
        fruit_rows = [r for r in rows if r["domain"] == "fruit"]
        summary["no_fruit_detected"] = sum("no_fruit_detected" in r["warnings"] for r in fruit_rows)
        summary["detected_box_classes"] = dict(
            Counter(c for r in fruit_rows for c in r["detection_classes"]).most_common()
        )
    else:
        summary["note"] = (
            "These photos are not tomato close-ups; the pipeline has no 'not a tomato' "
            "output, so every one of them still receives a leaf or fruit diagnosis."
        )
    return summary


def draw_overlay(image, result: dict):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width = max(3, img.width // 200)
    for det in result.get("detections") or []:
        draw.rectangle(det["bbox"], outline=(255, 40, 40), width=width)
        draw.text((det["bbox"][0] + 4, det["bbox"][1] + 4), f"{det['class']} {det['confidence']:.2f}", fill=(255, 255, 0))
    draw.text((10, 10), f"{result['domain']} {result['router_confidence']:.2f} | {result['class']} {result['confidence']:.2f}", fill=(255, 255, 0))
    img.thumbnail((1024, 1024))
    return img


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# External validation — field photos (inference only)",
        "",
        f"**Date:** {report['generated_at']}  ",
        f"**Models:** {', '.join(report['model_versions'].values())}  ",
        "**Data:** Google Drive validation folder (`data/external_validation/`, not in git). "
        "Validation only — nothing was trained or tuned on it. No disease labels, so accuracy "
        "cannot be computed; this report shows what the pipeline did.",
        "",
    ]
    for folder, s in report["summary"].items():
        lines += [f"## `{folder}` — expected: {s['expected']} (n={s['n_images']})", ""]
        lines += [f"- Routed: {s['routed']} (median router confidence {s['router_confidence_median']})"]
        if "routed_to_fruit_pct" in s:
            lines += [
                f"- Routed to fruit: **{s['routed_to_fruit_pct']}%**",
                f"- Fruit images with no YOLO detection: {s['no_fruit_detected']}",
                f"- Detected box classes: {s['detected_box_classes']}",
            ]
        lines += [
            f"- Predicted (primary) class: {s['predicted_class']}",
            f"- Median confidence {s['confidence_median']}; {s['low_confidence_lt_0_5']} below 0.5",
        ]
        if "note" in s:
            lines += [f"- {s['note']}"]
        lines += [""]
    lines += ["## Per image", "", "| Folder | Image | Routed (conf) | Class | Conf | Severity | Boxes |", "|---|---|---|---|---:|---:|---|"]
    for r in report["images"]:
        lines.append(
            f"| {r['folder'].split()[0]} | {r['image']} | {r['domain']} ({r['router_confidence']:.2f}) | "
            f"{r['class_display']} | {r['confidence']:.2f} | {r['severity_pct']:.1f} | "
            f"{', '.join(r['detection_classes']) or '—'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=ROOT / "data" / "external_validation")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "external_validation" / "report.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-overlays", action="store_true", help="Save annotated JPEGs next to the report.")
    args = parser.parse_args()

    out_json = args.out.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_json.parent / "overlays"
    pipeline = TomatoInferencePipeline(
        device=args.device, gradcam_dir=out_json.parent / "gradcam", with_advice=False
    )

    rows: list[dict] = []
    for folder_dir in sorted(p for p in args.dir.iterdir() if p.is_dir()):
        for path in sorted(p for p in folder_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            image = load_image(path)
            result = pipeline.infer_image(image, path)
            no_det = "no_fruit_detected" in result["warnings"]
            rows.append(
                {
                    "folder": folder_dir.name,
                    "image": path.name,
                    "domain": result["domain"],
                    "router_confidence": result["router_confidence"],
                    "class": result["class"],
                    "class_display": "No fruit detected" if no_det else result["class"],
                    "confidence": result["confidence"],
                    "severity_pct": result["severity_pct"],
                    "detection_classes": [d["class"] for d in result["detections"] or []],
                    "warnings": result["warnings"],
                }
            )
            if args.save_overlays:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                draw_overlay(image, result).save(overlay_dir / f"{folder_dir.name.split()[0]}_{path.stem}.jpg", quality=85)

    report = {
        "generated_at": str(date.today()),
        "purpose": "external validation only — never used for training or tuning",
        "model_versions": {
            "router": pipeline.router_model_version,
            "leaf": pipeline.leaf_model_version,
            "fruit": pipeline.fruit_model_version,
        },
        "summary": {
            folder: summarise([r for r in rows if r["folder"] == folder], FOLDER_EXPECTATION.get(folder, "unknown"))
            for folder in sorted({r["folder"] for r in rows})
        },
        "images": rows,
    }
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, out_json.with_suffix(".md"))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {out_json}\nSaved: {out_json.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
