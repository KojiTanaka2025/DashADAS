"""Highway-style lane marking detection and vanishing-point calibration.

Designed for clear dashed/solid white or yellow lane paint (motorway-like).
Near-horizontal edges (typical close guardrails / curbs) are rejected.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# Require a stretch of consistent lane hits before declaring calibration done.
MIN_GOOD_FRAMES = 20
# Vanishing-point scatter must stay within this fraction of the frame size.
MAX_VP_STD_FRAC = 0.03
# Intersection must sit in the upper band and near the horizontal center.
VP_Y_MAX_FRAC = 0.62
VP_X_CENTER_FRAC = 0.42


@dataclass
class LaneHit:
    vanishing_point: tuple[float, float]
    left_line: tuple[float, float, float, float]
    right_line: tuple[float, float, float, float]


def _line_from_points(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float] | None:
    """Return (slope, intercept) for y = slope * x + intercept, or None if vertical."""
    if abs(x2 - x1) < 1e-3:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _intersect(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float] | None:
    m1, b1 = left
    m2, b2 = right
    if abs(m1 - m2) < 1e-4:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


def _lane_mask(bgr: np.ndarray) -> np.ndarray:
    """White / yellow paint only — rejects most asphalt, sky, and dark guardrails."""
    # OpenCV HLS channel order is H, L, S (not H, S, L).
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hls, (0, 170, 0), (180, 255, 80))
    yellow = cv2.inRange(hsv, (15, 70, 120), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    # Prefer the road trapezoid: bottom center, not the far side rails.
    height, width = mask.shape
    poly = np.array(
        [
            [
                (int(width * 0.08), height - 1),
                (int(width * 0.42), int(height * 0.48)),
                (int(width * 0.58), int(height * 0.48)),
                (int(width * 0.92), height - 1),
            ]
        ],
        dtype=np.int32,
    )
    roi = np.zeros_like(mask)
    cv2.fillPoly(roi, poly, 255)
    mask = cv2.bitwise_and(mask, roi)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def detect_lane_hit(image: Image.Image | np.ndarray) -> LaneHit | None:
    """Find left+right lane markings and their vanishing point, or None."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = image
    height, width = bgr.shape[:2]
    mask = _lane_mask(bgr)
    edges = cv2.Canny(mask, 50, 150)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=max(30, int(height * 0.08)),
        maxLineGap=max(20, int(width * 0.04)),
    )
    if segments is None:
        return None

    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    left_seg: list[tuple[float, float, float, float]] = []
    right_seg: list[tuple[float, float, float, float]] = []

    for seg in segments[:, 0]:
        x1, y1, x2, y2 = map(float, seg)
        params = _line_from_points(x1, y1, x2, y2)
        if params is None:
            continue
        slope, intercept = params
        # Reject near-horizontal lines (curbs / close guardrails / horizon clutter).
        if abs(slope) < 0.35 or abs(slope) > 2.8:
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < height * 0.06:
            continue
        mid_x = 0.5 * (x1 + x2)
        # Image y grows downward: left lane slopes negative, right positive.
        if slope < 0 and mid_x < width * 0.55:
            left.append((slope, intercept))
            left_seg.append((x1, y1, x2, y2))
        elif slope > 0 and mid_x > width * 0.45:
            right.append((slope, intercept))
            right_seg.append((x1, y1, x2, y2))

    if len(left) < 1 or len(right) < 1:
        return None

    left_avg = (
        float(np.median([s for s, _ in left])),
        float(np.median([b for _, b in left])),
    )
    right_avg = (
        float(np.median([s for s, _ in right])),
        float(np.median([b for _, b in right])),
    )
    vp = _intersect(left_avg, right_avg)
    if vp is None:
        return None
    vx, vy = vp
    if not (0 <= vx < width and 0 <= vy < height * VP_Y_MAX_FRAC):
        return None
    if abs(vx - width * 0.5) > width * VP_X_CENTER_FRAC:
        return None

    def extend(line: tuple[float, float]) -> tuple[float, float, float, float]:
        m, b = line
        y1 = float(height - 1)
        y2 = float(max(vy, height * 0.45))
        x1 = (y1 - b) / m
        x2 = (y2 - b) / m
        return x1, y1, x2, y2

    return LaneHit(
        vanishing_point=(vx, vy),
        left_line=extend(left_avg),
        right_line=extend(right_avg),
    )


def draw_calibration_overlay(image: Image.Image, hit: LaneHit) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for line in (hit.left_line, hit.right_line):
        x1, y1, x2, y2 = [int(round(v)) for v in line]
        cv2.line(bgr, (x1, y1), (x2, y2), (80, 255, 214), 3)
    vx, vy = int(round(hit.vanishing_point[0])), int(round(hit.vanishing_point[1]))
    cv2.drawMarker(bgr, (vx, vy), (0, 215, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    cv2.circle(bgr, (vx, vy), 8, (0, 215, 255), 2)
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)


def calibrate_from_frames(frames: list[Image.Image]) -> dict:
    """Aggregate vanishing points across frames; complete only when stable."""
    if not frames:
        return {
            "calibrated": False,
            "message": "No frames available for calibration.",
            "good_frames": 0,
            "processed_frames": 0,
        }

    width, height = frames[0].size
    hits: list[LaneHit] = []
    last_hit: LaneHit | None = None
    last_image: Image.Image | None = None

    for image in frames:
        hit = detect_lane_hit(image)
        if hit is None:
            continue
        hits.append(hit)
        last_hit = hit
        last_image = image

    good = len(hits)
    if good < MIN_GOOD_FRAMES:
        return {
            "calibrated": False,
            "message": (
                f"Need clear highway-style lane markings on at least {MIN_GOOD_FRAMES} frames "
                f"(got {good}). Guardrails and curbs are ignored; try a straighter motorway clip."
            ),
            "good_frames": good,
            "processed_frames": len(frames),
            "image": {"width": width, "height": height},
        }

    xs = np.array([h.vanishing_point[0] for h in hits], dtype=np.float64)
    ys = np.array([h.vanishing_point[1] for h in hits], dtype=np.float64)
    std_x = float(xs.std())
    std_y = float(ys.std())
    max_std_x = width * MAX_VP_STD_FRAC
    max_std_y = height * MAX_VP_STD_FRAC
    if std_x > max_std_x or std_y > max_std_y:
        return {
            "calibrated": False,
            "message": (
                f"Lane vanishing point was unstable (std {std_x:.1f}x{std_y:.1f} px). "
                "Keep a straighter stretch of clear lane paint in view."
            ),
            "good_frames": good,
            "processed_frames": len(frames),
            "stability": {"std_x": round(std_x, 2), "std_y": round(std_y, 2)},
            "image": {"width": width, "height": height},
        }

    vx = float(np.median(xs))
    vy = float(np.median(ys))
    # Prefer a real frame overlay; synthesize lines through the median VP if needed.
    if last_hit is not None and last_image is not None:
        overlay_hit = LaneHit(
            vanishing_point=(vx, vy),
            left_line=last_hit.left_line,
            right_line=last_hit.right_line,
        )
        preview = draw_calibration_overlay(last_image, overlay_hit)
    else:
        preview = frames[-1]

    buffer = __import__("io").BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    preview_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "calibrated": True,
        "message": (
            f"Calibration complete from {good} stable highway-style lane frames. "
            "Session only — refresh clears it."
        ),
        "vanishing_point": {"x": round(vx, 1), "y": round(vy, 1)},
        "good_frames": good,
        "processed_frames": len(frames),
        "stability": {"std_x": round(std_x, 2), "std_y": round(std_y, 2)},
        "image": {"width": width, "height": height},
        "preview_jpeg_base64": preview_b64,
    }
