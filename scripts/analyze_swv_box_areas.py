"""Spotted_Wilt_Virus: box-area profile for detected vs missed val instances."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
SWV_ID = 7
IOU_THRES = 0.45
CONF_THRES = 0.25


def load_gt_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float) -> np.ndarray:
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return np.array([x1, y1, x2, y2])


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main() -> None:
    weights = REPO / "checkpoints" / "fruit_v1_yolov8n.pt"
    model = YOLO(str(weights))
    val_images = REPO / "data" / "splits" / "fruit" / "images" / "val"
    val_labels = REPO / "data" / "splits" / "fruit" / "labels" / "val"

    detected_areas: list[float] = []
    missed_areas: list[float] = []
    miscls: list[tuple[str, float, int]] = []

    for img_path in sorted(val_images.iterdir()):
        if not img_path.is_file():
            continue
        label_path = val_labels / f"{img_path.stem}.txt"
        gt_all = load_gt_boxes(label_path)
        gt_swv = [(i, b) for i, b in enumerate(gt_all) if b[0] == SWV_ID]
        if not gt_swv:
            continue

        preds = model.predict(
            str(img_path),
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False,
        )[0]
        pred_boxes = []
        if preds.boxes is not None and len(preds.boxes):
            xyxy = preds.boxes.xyxy.cpu().numpy()
            cls = preds.boxes.cls.cpu().numpy().astype(int)
            for box, pcls in zip(xyxy, cls):
                x1, y1, x2, y2 = box
                w = (x2 - x1) / preds.orig_shape[1]
                h = (y2 - y1) / preds.orig_shape[0]
                pred_boxes.append((pcls, np.array([x1, y1, x2, y2]), w * h))

        for _, (_, xc, yc, w, h) in gt_swv:
            area = w * h
            gt_xyxy = yolo_to_xyxy(xc, yc, w, h)
            gt_xyxy_px = gt_xyxy.copy()
            gt_xyxy_px[[0, 2]] *= preds.orig_shape[1]
            gt_xyxy_px[[1, 3]] *= preds.orig_shape[0]

            best_iou = 0.0
            best_cls = None
            for pcls, pxyxy, _ in pred_boxes:
                iou = box_iou(gt_xyxy_px, pxyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = pcls

            if best_iou >= IOU_THRES and best_cls == SWV_ID:
                detected_areas.append(area)
            elif best_iou >= IOU_THRES and best_cls is not None:
                names = list(yaml.safe_load(
                    (REPO / "data" / "splits" / "fruit" / "data.yaml").read_text()
                )["names"])
                miscls.append((names[best_cls], area, best_iou * 100))
                missed_areas.append(area)
            else:
                missed_areas.append(area)

    print("=== Spotted_Wilt_Virus val box-area analysis ===")
    print(f"GT instances on val: {len(detected_areas) + len(missed_areas)}")
    print(f"Detected (IoU>={IOU_THRES}, correct class): {len(detected_areas)}")
    print(f"Missed or wrong-class: {len(missed_areas)}")
    if detected_areas:
        print(
            f"Detected area (norm): mean={np.mean(detected_areas):.4f} "
            f"median={np.median(detected_areas):.4f} min={min(detected_areas):.4f}"
        )
    if missed_areas:
        print(
            f"Missed/wrong area (norm): mean={np.mean(missed_areas):.4f} "
            f"median={np.median(missed_areas):.4f} min={min(missed_areas):.4f}"
        )
    if miscls:
        print("Wrong-class matches (pred class, GT area, IoU%):")
        for pname, area, iou_pct in sorted(miscls, key=lambda x: -x[2]):
            print(f"  -> pred {pname}: area={area:.4f}, IoU={iou_pct:.1f}%")


if __name__ == "__main__":
    main()
