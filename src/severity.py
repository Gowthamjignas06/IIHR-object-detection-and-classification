"""Severity scoring: leaf (HSV/Lab segmentation) and fruit (box geometry)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class FruitDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def leaf_severity_pct(
    image: Image.Image,
    *,
    lower_green: tuple[int, int, int] = (25, 40, 40),
    upper_green: tuple[int, int, int] = (95, 255, 255),
    lab_deviation_threshold: float = 18.0,
) -> float:
    """Estimate diseased canopy fraction via HSV masking + Lab deviation within leaf region."""
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    canopy_mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green)) > 0
    canopy_pixels = int(canopy_mask.sum())
    if canopy_pixels == 0:
        return 0.0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    median_a = float(np.median(a_channel[canopy_mask]))
    median_b = float(np.median(b_channel[canopy_mask]))
    deviation = np.abs(a_channel - median_a) + np.abs(b_channel - median_b)
    lesion_mask = canopy_mask & (deviation > lab_deviation_threshold)
    return round(100.0 * float(lesion_mask.sum()) / canopy_pixels, 2)


def _union_box_area(boxes: list[tuple[float, float, float, float]]) -> float:
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return max((x2 - x1) * (y2 - y1), 1.0)


def _covered_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """Exact area covered by the union of boxes (overlaps counted once)."""
    xs = sorted({x for box in boxes for x in (box[0], box[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted(
            (y1, y2) for x1, y1, x2, y2 in boxes if x1 <= left and x2 >= right and y2 > y1
        )
        covered = 0.0
        cur_start, cur_end = None, None
        for y1, y2 in spans:
            if cur_end is None or y1 > cur_end:
                if cur_end is not None:
                    covered += cur_end - cur_start
                cur_start, cur_end = y1, y2
            else:
                cur_end = max(cur_end, y2)
        if cur_end is not None:
            covered += cur_end - cur_start
        total += covered * (right - left)
    return total


def fruit_severity_pct(
    detections: list[FruitDetection],
    image_size: tuple[int, int],
    *,
    healthy_class: str = "Healthy_Tomato",
) -> float:
    """Severity = area covered by disease boxes / estimated fruit area in the image."""
    width, height = image_size
    image_area = float(width * height)
    if not detections or image_area <= 0:
        return 0.0

    disease_boxes = [
        det.bbox_xyxy
        for det in detections
        if det.class_name != healthy_class
    ]
    if not disease_boxes:
        return 0.0

    healthy_boxes = [
        det.bbox_xyxy
        for det in detections
        if det.class_name == healthy_class
    ]
    all_boxes = [det.bbox_xyxy for det in detections]

    # Union, not sum: overlapping disease boxes must not be double-counted.
    disease_area = _covered_area(disease_boxes)

    if healthy_boxes:
        # Healthy detections localize the fruit body; disease boxes are fruit too,
        # so include them (otherwise severity can exceed 100% and get capped).
        estimated_fruit_area = max(_covered_area(healthy_boxes + disease_boxes), 1.0)
    elif len(all_boxes) > 1:
        # Multiple boxes: union of all detections approximates fruit extent.
        estimated_fruit_area = _union_box_area(all_boxes)
    else:
        # Single disease-only box: union would equal the box itself (always 100%).
        # Fall back to full image area per "estimated fruit area in the image".
        estimated_fruit_area = image_area

    return round(min(100.0, 100.0 * disease_area / estimated_fruit_area), 2)
