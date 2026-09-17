from __future__ import annotations

import numpy as np
from PIL import Image

PERSON_CLASS_ID = 0
INPUT_SIZE = 640


def letterbox(image: Image.Image, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, float, float]:
    src = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = src.shape[:2]
    scale = min(size / height, size / width)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    resized = np.asarray(image.resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    chw = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))
    return chw[np.newaxis, ...], scale, pad_x, pad_y


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou <= iou_thres]
    return keep


def decode_yolo_persons(
    raw: np.ndarray,
    image_size: tuple[int, int],
    scale: float,
    pad_x: float,
    pad_y: float,
    conf: float = 0.25,
    iou: float = 0.45,
) -> list[dict]:
    data = np.squeeze(raw)
    if data.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape {raw.shape}")
    if data.shape[0] < data.shape[1]:
        data = data.T
    # rows: [x, y, w, h, class...]
    boxes_xywh = data[:, :4]
    class_scores = data[:, 4:]
    person_scores = class_scores[:, PERSON_CLASS_ID] if class_scores.shape[1] > PERSON_CLASS_ID else class_scores.max(axis=1)
    mask = person_scores >= conf
    if not np.any(mask):
        return []
    boxes = _xywh_to_xyxy(boxes_xywh[mask])
    scores = person_scores[mask]
    keep = _nms(boxes, scores, iou)
    width, height = image_size
    detections = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        detections.append(
            {
                "label": "person",
                "score": round(float(scores[index]), 4),
                "bbox": [
                    round(float(max(0, min(width, x1))), 1),
                    round(float(max(0, min(height, y1))), 1),
                    round(float(max(0, min(width, x2))), 1),
                    round(float(max(0, min(height, y2))), 1),
                ],
            }
        )
    return detections
