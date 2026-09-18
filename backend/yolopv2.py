"""YOLOPv2 TorchScript runner (det + drivable + lane line).

Post-processing follows CAIC-AD/YOLOPv2 demo conventions (GPL-3.0 upstream).
Weights: https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

# Match the upstream LoadImages path: resize to 1280x720, then letterbox.
CANON_W, CANON_H = 1280, 720
IMG_SIZE = 640
STRIDE = 32

COCO_LABELS = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
DEFAULT_CLASS_IDS = (0, 2, 3, 5, 7)


@dataclass
class YoloPv2Result:
    detections: list[dict]
    lane_polylines: list[list[list[float]]]
    lane_mask: np.ndarray  # uint8 HxW on original image size, 0/255
    inference_ms: float
    model_name: str


def default_weights_path() -> Path:
    env = os.environ.get("DASHADAS_YOLOPV2_WEIGHTS", "").strip()
    if env:
        return Path(env)
    root = Path(__file__).resolve().parent.parent
    return root / "models" / "yolopv2.pt"


def weights_available(path: Path | None = None) -> bool:
    p = path or default_weights_path()
    return p.is_file() and p.stat().st_size > 1_000_000


def _letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (IMG_SIZE, IMG_SIZE),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    stride: int = STRIDE,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


def _make_grid(nx: int, ny: int, device: torch.device) -> torch.Tensor:
    yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def _split_for_trace_model(pred: list[torch.Tensor], anchor_grid: list[torch.Tensor]) -> torch.Tensor:
    z = []
    strides = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        pred[i] = pred[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = pred[i].sigmoid()
        gr = _make_grid(nx, ny, pred[i].device)
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + gr) * strides[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
        z.append(y.view(bs, -1, 85))
    return torch.cat(z, 1)


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list[int] | None = None,
) -> list[torch.Tensor]:
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 4096
    max_det = 300
    max_nms = 30000
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = _xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output


def _scale_coords(img1_shape: tuple[int, int], coords: torch.Tensor, img0_shape: tuple[int, ...]) -> torch.Tensor:
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


def _lane_line_mask(ll: torch.Tensor) -> np.ndarray:
    # Upstream crop assumes letterboxed 384x640-ish tensors from 1280x720 input.
    ll_predict = ll[:, :, 12:372, :]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode="bilinear")
    ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
    return ll_seg_mask.int().squeeze().cpu().numpy()


def _mask_to_polylines(mask: np.ndarray, max_contours: int = 8, epsilon_frac: float = 0.008) -> list[list[list[float]]]:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
    polylines: list[list[list[float]]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 40:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, True)
        pts = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in approx]
        if len(pts) >= 2:
            polylines.append(pts)
    return polylines


def _resize_mask_to_original(mask_canon: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """Map 1280x720 lane mask back to the original frame size."""
    if mask_canon.shape[0] != CANON_H or mask_canon.shape[1] != CANON_W:
        mask_canon = cv2.resize(mask_canon.astype(np.uint8), (CANON_W, CANON_H), interpolation=cv2.INTER_NEAREST)
    out = cv2.resize(mask_canon.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8) * 255


class YoloPv2Runner:
    def __init__(self, weights: Path | None = None, torch_device: str | None = None):
        self.weights = Path(weights) if weights else default_weights_path()
        if not weights_available(self.weights):
            raise FileNotFoundError(
                f"YOLOPv2 weights not found at {self.weights}. "
                "Download with scripts/download-yolopv2.sh"
            )
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(torch_device)
        self.half = self.torch_device.type == "cuda"
        self.model = torch.jit.load(str(self.weights), map_location=self.torch_device)
        self.model = self.model.to(self.torch_device)
        if self.half:
            self.model.half()
        self.model.eval()
        # Warmup with the common letterbox shape (384x640), not a square tensor.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 640, device=self.torch_device)
            if self.half:
                dummy = dummy.half()
            self.model(dummy)

    def to(self, torch_device: str) -> None:
        self.torch_device = torch.device(torch_device)
        self.half = self.torch_device.type == "cuda"
        self.model = self.model.to(self.torch_device)
        if self.half:
            self.model.half()
        else:
            self.model.float()

    @torch.no_grad()
    def infer(
        self,
        image: Image.Image,
        conf: float = 0.25,
        iou: float = 0.45,
        class_ids: tuple[int, ...] | list[int] = DEFAULT_CLASS_IDS,
    ) -> YoloPv2Result:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        orig_h, orig_w = rgb.shape[:2]
        bgr0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        im0 = cv2.resize(bgr0, (CANON_W, CANON_H), interpolation=cv2.INTER_LINEAR)
        img, _, _ = _letterbox(im0, new_shape=(IMG_SIZE, IMG_SIZE), auto=True, stride=STRIDE)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.torch_device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        started = time.perf_counter()
        (pred, anchor_grid), _seg, ll = self.model(tensor)
        pred = _split_for_trace_model(pred, anchor_grid)
        pred = _nms(pred, conf_thres=conf, iou_thres=iou, classes=list(class_ids))
        ll_mask_canon = _lane_line_mask(ll)
        inference_ms = (time.perf_counter() - started) * 1000.0

        detections: list[dict] = []
        det = pred[0]
        if det is not None and len(det):
            det = det.clone()
            det[:, :4] = _scale_coords(tensor.shape[2:], det[:, :4], im0.shape)
            sx = orig_w / float(CANON_W)
            sy = orig_h / float(CANON_H)
            for *xyxy, conf_v, cls in det.tolist():
                class_id = int(cls)
                x1, y1, x2, y2 = xyxy
                detections.append(
                    {
                        "label": COCO_LABELS.get(class_id, "object"),
                        "score": round(float(conf_v), 4),
                        "bbox": [
                            round(x1 * sx, 1),
                            round(y1 * sy, 1),
                            round(x2 * sx, 1),
                            round(y2 * sy, 1),
                        ],
                    }
                )

        lane_mask = _resize_mask_to_original(ll_mask_canon, orig_w, orig_h)
        polylines = _mask_to_polylines(lane_mask)
        return YoloPv2Result(
            detections=detections,
            lane_polylines=polylines,
            lane_mask=lane_mask,
            inference_ms=round(inference_ms, 1),
            model_name="yolopv2.pt",
        )
