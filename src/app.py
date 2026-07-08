"""
Tomato Disease Diagnosis — Gradio web interface (full pipeline).

Classifies any tomato image into one of 10 leaf disease classes
or one of 8 fruit defect classes, with:
  - Grad-CAM heatmap for leaf images
  - YOLO bounding box overlay for fruit images
  - Severity percentage
  - Annotated result image saved to results/inference_results/
  - Running JSON log

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

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infer_pipeline import TomatoInferencePipeline

# ── Output directories ────────────────────────────────────────────────────────
_RESULTS_DIR = _ROOT / "results" / "inference_results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _RESULTS_DIR / "inference_log.json"

# ── Colour palette — leaf classes ─────────────────────────────────────────────
_LEAF_COLORS: dict[str, tuple[int, int, int]] = {
    "Bacterial spot":                  (200, 120,  40),
    "Early blight":                    (220,  80,  40),
    "Healthy":                         ( 40, 170,  60),
    "Late blight":                     (140,  40,  40),
    "Leaf Mold":                       (160, 100,  40),
    "Septoria leaf spot":              (200, 160,  40),
    "Spider mites (Two-spotted)":      (180, 180,  40),
    "Target Spot":                     (200,  80,  80),
    "Tomato Yellow Leaf Curl Virus":   (220, 200,  40),
    "Tomato mosaic virus":             (160,  40, 160),
}

# ── Colour palette — fruit classes ────────────────────────────────────────────
_FRUIT_COLORS: dict[str, tuple[int, int, int]] = {
    "Anthracnose":        (200, 140,  30),
    "Blossom_End_Rot":    (180,  50, 180),
    "Catfaced":           (180, 100,  50),
    "Fruit_Cracking":     (160, 160,  40),
    "Healthy_Tomato":     ( 30, 140, 200),
    "Late_Blight":        (160,  60,  60),
    "Mold":               (100, 160, 100),
    "Spotted_Wilt_Virus": (100, 100, 200),
}

_DEFAULT_COLOR = (120, 120, 120)

# ── Singleton pipeline ────────────────────────────────────────────────────────
_pipeline: TomatoInferencePipeline | None = None


def _get_pipeline() -> TomatoInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TomatoInferencePipeline(
            root=_ROOT,
            gradcam_dir=_RESULTS_DIR / "gradcam",
        )
    return _pipeline


# ── YOLO detection overlay ────────────────────────────────────────────────────

def _draw_fruit_detections(
    original: Image.Image,
    detections: list[dict] | None,
) -> Image.Image:
    img = original.convert("RGB").copy()
    if not detections:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "No detections", fill=(200, 200, 200))
        return img

    draw = ImageDraw.Draw(img)
    try:
        font    = ImageFont.truetype("arial.ttf", 16)
        font_sm = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font    = ImageFont.load_default()
        font_sm = font

    for det in detections:
        cls   = det["class"]
        conf  = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = _FRUIT_COLORS.get(cls, _DEFAULT_COLOR)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls}  {conf:.2f}"
        try:
            tb = draw.textbbox((x1, y1 - 20), label, font=font_sm)
            draw.rectangle(tb, fill=color)
        except AttributeError:
            draw.rectangle([x1, y1 - 20, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 19), label, fill=(255, 255, 255), font=font_sm)

    return img


# ── Annotated score panel ─────────────────────────────────────────────────────

def _build_score_panel(
    original: Image.Image,
    result: dict,
    extra_label: str,
    class_colors: dict[str, tuple[int, int, int]],
) -> Image.Image:
    W, H = original.size
    panel_w = max(360, int(W * 0.6))
    canvas  = Image.new("RGB", (W + panel_w, H), (20, 20, 24))
    canvas.paste(original, (0, 0))

    draw      = ImageDraw.Draw(canvas)
    cls_name  = result["class"]
    confidence = result["confidence"]
    severity   = result.get("severity_pct", 0.0)
    color      = class_colors.get(cls_name, _DEFAULT_COLOR)

    draw.rectangle([W, 0, W + panel_w, H], fill=(28, 28, 34))

    try:
        font_lg = ImageFont.truetype("arial.ttf", 24)
        font_md = ImageFont.truetype("arial.ttf", 17)
        font_sm = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font_lg = ImageFont.load_default()
        font_md = font_lg
        font_sm = font_lg

    px, py = W + 16, 18

    draw.text((px, py), "DIAGNOSTIC RESULT", fill=(160, 160, 165), font=font_sm)
    py += 20
    draw.text((px, py), f"({extra_label})", fill=(100, 100, 110), font=font_sm)
    py += 26

    for line in textwrap.wrap(cls_name, width=20):
        draw.text((px, py), line, fill=color, font=font_lg)
        py += 30
    py += 4

    draw.text((px, py), f"Confidence: {confidence * 100:.1f}%", fill=(210, 210, 210), font=font_md)
    py += 24

    bar_w, bar_h = panel_w - 32, 14
    draw.rectangle([px, py, px + bar_w, py + bar_h], fill=(55, 55, 62))
    draw.rectangle([px, py, px + int(bar_w * confidence), py + bar_h], fill=color)
    py += bar_h + 10

    draw.text((px, py), f"Severity: {severity:.1f}%", fill=(180, 180, 185), font=font_sm)
    py += 20

    draw.line([px, py, W + panel_w - 16, py], fill=(55, 55, 65), width=1)
    py += 12

    domain_label = "ALL CLASSES (leaf)" if result["domain"] == "leaf" else "ALL DETECTIONS (fruit)"
    draw.text((px, py), domain_label, fill=(130, 130, 140), font=font_sm)
    py += 20

    # Build display rows
    if result["domain"] == "leaf":
        # Show top-6 leaf classes by score (not all 10 — panel space)
        # We don't have full softmax here (pipeline returns only top class),
        # so we show the single prediction prominently + severity info.
        draw.text((px, py), f"▶  {cls_name}", fill=color, font=font_sm)
        py += 18
        fill_bar = int((panel_w - 32) * confidence)
        draw.rectangle([px, py, px + panel_w - 32, py + 7], fill=(50, 50, 58))
        draw.rectangle([px, py, px + fill_bar, py + 7], fill=color)
        draw.text((W + panel_w - 60, py - 2), f"{confidence:.4f}", fill=(190, 190, 195), font=font_sm)
        py += 20
    else:
        detections = result.get("detections") or []
        shown = detections[:8]  # max 8 rows
        for det in shown:
            dc     = det["class"]
            dconf  = det["confidence"]
            dc_col = _FRUIT_COLORS.get(dc, _DEFAULT_COLOR)
            prefix = "▶  " if dc == cls_name else "   "
            draw.text((px, py), f"{prefix}{dc}", fill=dc_col, font=font_sm)
            fill_mini = int((panel_w - 32) * dconf)
            mini_y = py + 17
            draw.rectangle([px, mini_y, px + panel_w - 32, mini_y + 6], fill=(50, 50, 58))
            draw.rectangle([px, mini_y, px + fill_mini, mini_y + 6], fill=dc_col)
            draw.text((W + panel_w - 60, py), f"{dconf:.3f}", fill=(185, 185, 190), font=font_sm)
            py += 34

    draw.line([px, py, W + panel_w - 16, py], fill=(55, 55, 65), width=1)
    py += 12
    draw.text((px, py), f"Domain: {result['domain']}  |  Model: {result['model_version']}", fill=(110, 110, 118), font=font_sm)
    py += 18
    draw.text((px, py), datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), fill=(90, 90, 98), font=font_sm)

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

def diagnose_image(
    image: Image.Image,
) -> tuple[Image.Image | None, Image.Image | None, str, str]:
    """
    Called by Gradio on every image upload.

    Returns
    -------
    explain_img  : Grad-CAM overlay (leaf) or YOLO box overlay (fruit)
    panel_img    : original + dark annotation panel
    md           : markdown result summary
    saved_msg    : saved file paths
    """
    if image is None:
        return None, None, "No image provided.", ""

    rgb = image.convert("RGB")

    # Save to temp file — pipeline needs a real path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    rgb.save(tmp_path, format="JPEG", quality=95)

    try:
        pipeline = _get_pipeline()
        result   = pipeline.infer(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    domain     = result["domain"]
    cls_name   = result["class"]
    confidence = result["confidence"]
    severity   = result.get("severity_pct", 0.0)

    # ── Explainability image ───────────────────────────────────────────────────
    if domain == "leaf":
        gradcam_path = result.get("gradcam_overlay")
        if gradcam_path and Path(gradcam_path).exists():
            explain_img = Image.open(gradcam_path).convert("RGB")
        else:
            explain_img = rgb.copy()
        extra_label  = "Grad-CAM heatmap"
        class_colors = _LEAF_COLORS
    else:
        explain_img  = _draw_fruit_detections(rgb, result.get("detections"))
        extra_label  = "YOLO detections"
        class_colors = _FRUIT_COLORS

    # ── Score panel ────────────────────────────────────────────────────────────
    panel_img = _build_score_panel(rgb, result, extra_label, class_colors)

    # ── Save results ───────────────────────────────────────────────────────────
    ts_str     = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_class = cls_name.replace(" ", "_").replace("(", "").replace(")", "")
    panel_path   = _RESULTS_DIR / f"{ts_str}_{safe_class}_result.jpg"
    explain_path = _RESULTS_DIR / f"{ts_str}_{safe_class}_explain.jpg"
    panel_img.save(panel_path,   format="JPEG", quality=95)
    explain_img.save(explain_path, format="JPEG", quality=95)

    log_entry = {
        "timestamp":     datetime.now().isoformat(),
        "result_image":  str(panel_path),
        "explain_image": str(explain_path),
        **result,
    }
    _append_log(log_entry)

    # ── Markdown summary ───────────────────────────────────────────────────────
    severity_note = (
        f"Lesion area: **{severity:.1f}%** of leaf canopy"
        if domain == "leaf"
        else f"Affected area: **{severity:.1f}%** of fruit"
    )

    detections = result.get("detections") or []
    det_rows   = "\n".join(
        f"| {d['class']} | {d['confidence']:.3f} | {d['bbox']} |"
        for d in detections
    ) if detections else ""

    det_table = (
        f"\n\n**All detections:**\n\n| Class | Conf | BBox |\n|---|---:|---|\n{det_rows}"
        if det_rows else ""
    )

    md = "\n".join([
        f"### {cls_name}  —  {confidence * 100:.1f}% confident",
        f"Domain: **{domain}**  |  {severity_note}  |  Model: `{result['model_version']}`",
        det_table,
    ])

    saved_msg = (
        f"Saved:\n"
        f"  Result panel  → {panel_path.name}\n"
        f"  {extra_label.title()} → {explain_path.name}\n"
        f"  Log → inference_log.json"
    )

    return explain_img, panel_img, md, saved_msg


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_LEAF_CLASSES = [
    "Bacterial spot", "Early blight", "Healthy", "Late blight",
    "Leaf Mold", "Septoria leaf spot", "Spider mites (Two-spotted)",
    "Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato mosaic virus",
]
_FRUIT_CLASSES = [
    "Anthracnose", "Blossom_End_Rot", "Catfaced", "Fruit_Cracking",
    "Healthy_Tomato", "Late_Blight", "Mold", "Spotted_Wilt_Virus",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Tomato Disease Diagnosis",
        theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
        css="""
        .title { font-size: 1.55em; font-weight: 700; margin-bottom: 3px; }
        .sub   { color: #999; font-size: 0.95em; margin-bottom: 16px; }
        """,
    ) as demo:

        gr.HTML("""
            <div class="title">Tomato Disease Diagnosis</div>
            <div class="sub">
              Upload any tomato leaf or fruit photo.
              Routes automatically — leaf → 10-class classifier + Grad-CAM,
              fruit → 8-class detector + bounding boxes.
            </div>
        """)

        with gr.Row():
            # ── Upload ──────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                inp = gr.Image(
                    type="pil",
                    label="Upload tomato image",
                    image_mode="RGB",
                    height=320,
                )
                btn = gr.Button("Diagnose", variant="primary", size="lg")

                with gr.Accordion("Leaf classes (10)", open=False):
                    gr.Markdown("\n".join(f"- {c}" for c in _LEAF_CLASSES))
                with gr.Accordion("Fruit classes (8)", open=False):
                    gr.Markdown("\n".join(f"- {c}" for c in _FRUIT_CLASSES))

            # ── Explainability ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                out_explain = gr.Image(
                    label="Grad-CAM heatmap (leaf)  /  YOLO detections (fruit)",
                    type="pil",
                    height=320,
                )
                gr.Markdown(
                    "*Leaf:* red/yellow = region the model focused on.  \n"
                    "*Fruit:* coloured boxes around each detected defect."
                )

            # ── Score panel ──────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                out_panel = gr.Image(
                    label="Result panel",
                    type="pil",
                    height=320,
                )

        with gr.Row():
            out_md = gr.Markdown()
        with gr.Row():
            out_saved = gr.Textbox(
                label="Saved files",
                interactive=False,
                lines=3,
            )

        outputs = [out_explain, out_panel, out_md, out_saved]
        btn.click(fn=diagnose_image, inputs=[inp], outputs=outputs)
        inp.change(fn=diagnose_image, inputs=[inp], outputs=outputs)

        gr.Markdown("""
---
All results auto-saved to `results/inference_results/`  
— result panel JPEG + Grad-CAM/detection JPEG + running `inference_log.json`.
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
