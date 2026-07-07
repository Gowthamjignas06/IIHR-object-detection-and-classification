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
import tempfile
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
from gradcam import compute_gradcam, overlay_gradcam_on_image

# ── Output directory ──────────────────────────────────────────────────────────
_RESULTS_DIR = _ROOT / "results" / "inference_results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _RESULTS_DIR / "inference_log.json"

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
        draw.text((10, 10), "No detections", fill=(200, 200, 200))
        return img

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_sm = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

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
    color = _CLASS_COLORS.get(predicted, _DEFAULT_COLOR)

    draw.rectangle([W, 0, total_w, H], fill=(28, 28, 34))

    try:
        font_large = ImageFont.truetype("arial.ttf", 26)
        font_med   = ImageFont.truetype("arial.ttf", 17)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font_large = ImageFont.load_default()
        font_med   = font_large
        font_small = font_large

    px = W + 16
    py = 18

    draw.text((px, py), "DIAGNOSTIC RESULT", fill=(160, 160, 160), font=font_small)
    py += 22
    draw.text((px, py), f"({extra_label})", fill=(100, 100, 110), font=font_small)
    py += 26

    for line in textwrap.wrap(predicted, width=18):
        draw.text((px, py), line, fill=color, font=font_large)
        py += 32
    py += 4

    conf_pct = int(result["confidence"] * 100)
    draw.text((px, py), f"Confidence: {conf_pct}%", fill=(210, 210, 210), font=font_med)
    py += 26

    bar_w = panel_w - 32
    bar_h = 16
    draw.rectangle([px, py, px + bar_w, py + bar_h], fill=(55, 55, 62))
    draw.rectangle([px, py, px + int(bar_w * result["confidence"]), py + bar_h], fill=color)
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
        prefix = "▶ " if is_pred else "  "
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
) -> tuple[Image.Image | None, Image.Image | None, str, str]:
    """
    Called by Gradio on every image upload / classify click.

    Returns
    -------
    explainability_img : Grad-CAM overlay (leaf) or YOLO box overlay (fruit)
    result_panel_img   : original + dark score panel
    md                 : markdown score table
    saved_msg          : path of saved result file
    """
    if image is None:
        return None, None, "No image provided.", ""

    rgb = image.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    rgb.save(tmp_path, format="JPEG", quality=95)

    try:
        clf = _get_classifier()
        result = clf.classify(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

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
    safe_class = result["predicted_class"].replace(" ", "_")
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

    md = "\n".join([
        f"### {result['predicted_class']}  —  {result['confidence'] * 100:.1f}% confident",
        f"Domain: **{result['domain']}**  |  "
        f"Router confidence: **{result['router_confidence']:.4f}**",
        "",
        "| Class | Score | Bar |",
        "|---|---:|---|",
        *rows,
    ])

    saved_msg = (
        f"Saved:\n"
        f"  Result panel  → {panel_path.name}\n"
        f"  {extra_label.title()} → {explain_path.name}\n"
        f"  Log → inference_log.json"
    )

    return explain_img, panel_img, md, saved_msg


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Four-Class Tomato Diagnostic",
        theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
        css="""
        .title  { font-size: 1.55em; font-weight: 700; margin-bottom: 3px; }
        .sub    { color: #999; margin-bottom: 16px; font-size: 0.95em; }
        """,
    ) as demo:

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
            out_saved = gr.Textbox(
                label="Saved files",
                interactive=False,
                lines=3,
            )

        # ── Wire buttons ───────────────────────────────────────────────────────
        outputs = [out_explain, out_panel, out_scores, out_saved]
        btn.click(fn=classify_image, inputs=[inp], outputs=outputs)
        inp.change(fn=classify_image, inputs=[inp], outputs=outputs)

        gr.Markdown("""
---
All results auto-saved to `results/inference_results/`  
— result panel JPEG + explain JPEG + running `inference_log.json` per run.
""")

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
