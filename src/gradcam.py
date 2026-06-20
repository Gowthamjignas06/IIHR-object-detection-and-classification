"""Grad-CAM explainability for the leaf classifier (EfficientNet-B0)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _target_layer(model: torch.nn.Module) -> torch.nn.Module:
    return model.features[-1]


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap in [0, 1], shape (H, W)."""
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inp, out):
        activations["value"] = out

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    layer = _target_layer(model)
    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        score = logits[0, target_class]
        score.backward()
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations["value"]).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        fh.remove()
        bh.remove()


def overlay_gradcam_on_image(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(rgb, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(blended)


def save_gradcam_overlay(
    image: Image.Image,
    cam: np.ndarray,
    output_path: Path | str,
    alpha: float = 0.45,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_gradcam_on_image(image, cam, alpha=alpha).save(path)
    return str(path.resolve())
