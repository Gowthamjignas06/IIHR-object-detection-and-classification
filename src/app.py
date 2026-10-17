"""
Four-class tomato diagnostic — Gradio web interface.

Leaf images: shows original + Grad-CAM heatmap (highlights which region
             drove the prediction) + annotated score panel.
Fruit images: shows original with YOLO bounding boxes drawn on it
              + annotated score panel.

Run:
    python src/app.py
Then open http://127.0.0.1:7860 in your browser.
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from four_class_infer import (
    FOUR_CLASSES,
    FourClassDiagnosticClassifier,
    _LEAF_EARLY_BLIGHT_IDX,
    _LEAF_HEALTHY_IDX,
)
from advice import advice_to_markdown
from gradcam import compute_gradcam, overlay_gradcam_on_image

# ── Output directory ──────────────────────────────────────────────────────────
_RESULTS_DIR = _ROOT / "results" / "inference_results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# New runs log here (gitignored); the committed inference_log.json is a frozen v1 artifact.
_LOG_FILE = _RESULTS_DIR / "app_log.json"

# ── Colour palette ────────────────────────────────────────────────────────────
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Early blight":    (220, 80,  40),
    "Blossom_End_Rot": (180, 50,  180),
    "Healthy leaf":    (40,  170, 60),
    "Healthy plant":   (30,  140, 200),
}
# Colours for YOLO box labels (BGR for cv2, converted to RGB for PIL)
_YOLO_BOX_COLOR: dict[str, tuple[int, int, int]] = {
    "Blossom_End_Rot": (180, 50,  180),
    "Healthy_Tomato":  (30,  140, 200),
    "Anthracnose":     (200, 140, 30),
    "Catfaced":        (180, 100, 50),
    "Fruit_Cracking":  (160, 160, 40),
    "Late_Blight":     (160, 60,  60),
    "Mold":            (100, 160, 100),
    "Spotted_Wilt_Virus": (100, 100, 200),
}
_DEFAULT_COLOR = (120, 120, 120)
_GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
_THEME = gr.themes.Base(primary_hue="green", neutral_hue="slate")
_CSS = """
        .title  { font-size: 1.55em; font-weight: 700; margin-bottom: 3px; }
        .sub    { color: #999; margin-bottom: 16px; font-size: 0.95em; }
        """


def _font(size: int) -> ImageFont.ImageFont:
    """Arial where available (Windows/macOS); Pillow's scalable default otherwise."""
    for name in ("arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def _display_label(result: dict) -> tuple[str, float]:
    """What to show users. JSON keeps predicted_class unchanged for compatibility,
    but "Healthy plant 1.0" can mean "nothing found" or "a non-target condition"."""
    warnings = result.get("warnings") or []
    if "no_fruit_detected" in warnings:
        return "No fruit detected", 0.0
    others = result.get("other_conditions") or []
    if others and ("other_condition_detected" in warnings or "other_leaf_condition_detected" in warnings):
        return f"Other condition: {others[0]['class']}", float(others[0]["confidence"])
    return result["predicted_class"], float(result["confidence"])

# ── Singleton classifier ───────────────────────────────────────────────────────
_clf: FourClassDiagnosticClassifier | None = None


def _get_classifier() -> FourClassDiagnosticClassifier:
    global _clf
    if _clf is None:
        _clf = FourClassDiagnosticClassifier(root=_ROOT)
    return _clf


# ── Grad-CAM for leaf images ───────────────────────────────────────────────────

def _compute_leaf_gradcam(
    clf: FourClassDiagnosticClassifier,
    original_image: Image.Image,
    predicted_class: str,
) -> Image.Image:
    """
    Run Grad-CAM on the leaf model for the predicted class and return a
    PIL image of the heatmap blended over the original.
    """
    leaf_idx = (
        _LEAF_EARLY_BLIGHT_IDX
        if predicted_class == "Early blight"
        else _LEAF_HEALTHY_IDX
    )
    cam_tensor = (
        clf._transform(original_image)
        .unsqueeze(0)
        .to(clf.device)
        .requires_grad_(True)
    )
    clf._leaf_model.eval()
    cam = compute_gradcam(clf._leaf_model, cam_tensor, target_class=leaf_idx)
    return overlay_gradcam_on_image(original_image, cam, alpha=0.5)


# ── YOLO detection overlay for fruit images ────────────────────────────────────

def _draw_fruit_detections(
    original_image: Image.Image,
    detections: list[dict] | None,
) -> Image.Image:
    """
    Draw YOLO bounding boxes and class labels on a copy of the original image.
    Boxes for the two four-class targets are thicker and colour-coded.
    """
    img = original_image.convert("RGB").copy()
    if not detections:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "No fruit detected", fill=(230, 230, 230), font=_font(18))
        return img

    draw = ImageDraw.Draw(img)
    font_sm = _font(13)

    for det in detections:
        cls_name = det["class"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        # Map Healthy_Tomato → Healthy plant for display
        display_name = "Healthy plant" if cls_name == "Healthy_Tomato" else cls_name
        color = _YOLO_BOX_COLOR.get(cls_name, _DEFAULT_COLOR)

        # Thicker box for the four-class targets
        is_target = cls_name in ("Blossom_End_Rot", "Healthy_Tomato")
        width = 3 if is_target else 1

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label = f"{display_name}  {conf:.2f}"
        # Label background
        try:
            bbox_text = draw.textbbox((x1, y1 - 20), label, font=font_sm)
            draw.rectangle(bbox_text, fill=color)
        except AttributeError:
            draw.rectangle([x1, y1 - 20, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 19), label, fill=(255, 255, 255), font=font_sm)

    return img


# ── Score panel (right-side annotation) ───────────────────────────────────────

def _build_score_panel(
    original: Image.Image,
    result: dict,
    extra_label: str,
) -> Image.Image:
    """
    Paste the original image on the left and a dark score panel on the right.
    `extra_label` is a short string shown at the top of the panel describing
    what the middle image shows (e.g. "Grad-CAM" or "YOLO detections").
    """
    W, H = original.size
    panel_w = max(340, W // 2)
    total_w = W + panel_w
    canvas = Image.new("RGB", (total_w, H), (20, 20, 24))
    canvas.paste(original, (0, 0))

    draw = ImageDraw.Draw(canvas)
    predicted = result["predicted_class"]
    headline, headline_conf = _display_label(result)
    color = _CLASS_COLORS.get(headline, _DEFAULT_COLOR)

    draw.rectangle([W, 0, total_w, H], fill=(28, 28, 34))

    font_large = _font(26)
    font_med   = _font(17)
    font_small = _font(14)

    px = W + 16
    py = 18

    draw.text((px, py), "DIAGNOSTIC RESULT", fill=(160, 160, 160), font=font_small)
    py += 22
    draw.text((px, py), f"({extra_label})", fill=(100, 100, 110), font=font_small)
    py += 26

    for line in textwrap.wrap(headline, width=18):
        draw.text((px, py), line, fill=color, font=font_large)
        py += 32
    py += 4

    conf_pct = int(headline_conf * 100)
    draw.text((px, py), f"Confidence: {conf_pct}%", fill=(210, 210, 210), font=font_med)
    py += 26

    bar_w = panel_w - 32
    bar_h = 16
    draw.rectangle([px, py, px + bar_w, py + bar_h], fill=(55, 55, 62))
    draw.rectangle([px, py, px + int(bar_w * headline_conf), py + bar_h], fill=color)
    py += bar_h + 20

    draw.line([px, py, W + panel_w - 16, py], fill=(60, 60, 68), width=1)
    py += 12

    draw.text((px, py), "ALL CLASS SCORES", fill=(130, 130, 140), font=font_small)
    py += 20

    scores = result["class_scores"]
    max_score = max(scores.values()) if max(scores.values()) > 0 else 1.0

    for cls in FOUR_CLASSES:
        score = scores.get(cls, 0.0)
        c = _CLASS_COLORS.get(cls, _DEFAULT_COLOR)
        is_pred = cls == predicted
        prefix = "» " if is_pred else "  "
        draw.text(
            (px, py),
            f"{prefix}{cls}",
            fill=c if is_pred else (170, 170, 175),
            font=font_small,
        )
        draw.text(
            (W + panel_w - 56, py),
            f"{score:.4f}",
            fill=(190, 190, 195),
            font=font_small,
        )
        mini_y = py + 17
        mini_w = panel_w - 32
        draw.rectangle([px, mini_y, px + mini_w, mini_y + 7], fill=(50, 50, 58))
        if max_score > 0 and score > 0:
            draw.rectangle(
                [px, mini_y, px + int(mini_w * score / max_score), mini_y + 7],
                fill=c,
            )
        py += 38

    py += 4
    draw.line([px, py, W + panel_w - 16, py], fill=(60, 60, 68), width=1)
    py += 12
    draw.text(
        (px, py),
        f"Domain: {result['domain']}  |  Router: {result['router_confidence']:.4f}",
        fill=(110, 110, 118),
        font=font_small,
    )
    py += 18
    draw.text(
        (px, py),
        datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
        fill=(90, 90, 98),
        font=font_small,
    )
    return canvas


# ── Log helper ────────────────────────────────────────────────────────────────

def _append_log(entry: dict) -> None:
    log: list[dict] = []
    if _LOG_FILE.exists():
        try:
            log = json.loads(_LOG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log = []
    log.append(entry)
    _LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")


# ── Core inference function ───────────────────────────────────────────────────

def classify_image(
    image: Image.Image,
) -> tuple[Image.Image | None, Image.Image | None, str, str, str]:
    """
    Called by Gradio on every image upload / classify click.

    Returns
    -------
    explainability_img : Grad-CAM overlay (leaf) or YOLO box overlay (fruit)
    result_panel_img   : original + dark score panel
    md                 : markdown score table
    saved_msg          : path of saved result file
    advice_md          : disease explanation, treatment plan and action steps
    """
    if image is None:
        return None, None, "No image provided.", "", ""

    rgb = image.convert("RGB")

    # In-memory inference: re-encoding to a temp JPEG shifted scores vs the CLI.
    clf = _get_classifier()
    result = clf.classify_image(rgb)

    # ── Explainability image ───────────────────────────────────────────────────
    if result["domain"] == "leaf":
        explain_img = _compute_leaf_gradcam(clf, rgb, result["predicted_class"])
        extra_label = "Grad-CAM heatmap"
    else:
        explain_img = _draw_fruit_detections(rgb, result.get("detections"))
        extra_label = "YOLO detections"

    # ── Annotated score panel (original + dark panel) ─────────────────────────
    panel_img = _build_score_panel(rgb, result, extra_label)

    # ── Save result images ────────────────────────────────────────────────────
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    headline, headline_conf = _display_label(result)
    safe_class = "".join(ch if ch.isalnum() else "_" for ch in headline).strip("_")
    panel_path = _RESULTS_DIR / f"{ts_str}_{safe_class}_result.jpg"
    explain_path = _RESULTS_DIR / f"{ts_str}_{safe_class}_explain.jpg"
    panel_img.save(panel_path, format="JPEG", quality=95)
    explain_img.save(explain_path, format="JPEG", quality=95)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "result_image": str(panel_path),
        "explain_image": str(explain_path),
        **result,
    }
    _append_log(log_entry)

    # ── Markdown score table ───────────────────────────────────────────────────
    scores = result["class_scores"]
    rows = []
    for cls in FOUR_CLASSES:
        score = scores.get(cls, 0.0)
        bar = "█" * int(score * 20)
        marker = " ◀" if cls == result["predicted_class"] else ""
        rows.append(f"| {cls} | {score:.4f} | `{bar:<20}` |{marker}")

    status = ""
    if headline != result["predicted_class"]:
        status = (
            f"_Four-class output (unchanged for compatibility): **{result['predicted_class']}** "
            f"{result['confidence']:.2f}. Warnings: {', '.join(result.get('warnings') or [])}._"
        )
    md = "\n".join([
        f"### {headline}  —  {headline_conf * 100:.1f}% confident",
        f"Domain: **{result['domain']}**  |  "
        f"Router confidence: **{result['router_confidence']:.4f}**  |  "
        f"Severity: **{result.get('severity_pct', 0.0):.1f}%**",
        status,
        "",
        "| Class | Score | Bar |",
        "|---|---:|---|",
        *rows,
    ])

    saved_msg = (
        f"Saved:\n"
        f"  Result panel  → {panel_path.name}\n"
        f"  {extra_label.title()} → {explain_path.name}\n"
        f"  Log → {_LOG_FILE.name}"
    )

    advice_md = advice_to_markdown(result.get("advice") or {})
    return explain_img, panel_img, md, saved_msg, advice_md


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    # Gradio 6 moved theme/css from Blocks() to launch(); older versions need them here.
    style_kwargs = {} if _GRADIO_6 else {"theme": _THEME, "css": _CSS}
    with gr.Blocks(title="Four-Class Tomato Diagnostic", **style_kwargs) as demo:

        gr.HTML("""
            <div class="title">Four-Class Tomato Diagnostic</div>
            <div class="sub">
              Upload a tomato leaf or fruit photo. The model classifies it and
              explains its reasoning — Grad-CAM for leaves, detection boxes for fruit.
            </div>
        """)

        with gr.Row():
            # ── Left: upload ────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                inp = gr.Image(
                    type="pil",
                    label="Upload tomato image",
                    image_mode="RGB",
                    height=320,
                )
                btn = gr.Button("Classify", variant="primary", size="lg")
                gr.Markdown("""
**Target classes:**
- 🍂 **Early blight** *(leaf disease)*
- 🟣 **Blossom_End_Rot** *(fruit disorder)*
- 🌿 **Healthy leaf**
- 🍅 **Healthy plant**
""")

            # ── Middle: Grad-CAM / detection overlay ─────────────────────────
            with gr.Column(scale=1, min_width=320):
                out_explain = gr.Image(
                    label="Grad-CAM heatmap (leaf)  /  YOLO detections (fruit)",
                    type="pil",
                    height=320,
                )
                gr.Markdown("""
*Leaf images:* red/yellow regions show what the model focused on.  
*Fruit images:* coloured boxes show every detected defect.
""")

            # ── Right: annotated score panel ──────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                out_panel = gr.Image(
                    label="Result panel",
                    type="pil",
                    height=320,
                )

        # ── Score table + saved path ───────────────────────────────────────────
        with gr.Row():
            out_scores = gr.Markdown()
        with gr.Row():
            out_advice = gr.Markdown(label="Disease guide")
        with gr.Row():
            out_saved = gr.Textbox(
                label="Saved files",
                interactive=False,
                lines=3,
            )

        # ── Wire buttons ───────────────────────────────────────────────────────
        outputs = [out_explain, out_panel, out_scores, out_saved, out_advice]
        # Button only: also binding inp.change ran every upload twice (and logged twice).
        btn.click(fn=classify_image, inputs=[inp], outputs=outputs)

        gr.Markdown("""
---
All results auto-saved to `results/inference_results/`  
— result panel JPEG + explain JPEG + running `app_log.json` (all gitignored).
""")

    return demo


def main() -> None:
    demo = build_ui()
    style_kwargs = {"theme": _THEME, "css": _CSS} if _GRADIO_6 else {}
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        **style_kwargs,
    )


if __name__ == "__main__":
    main()
