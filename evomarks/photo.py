"""Real photo -> aligned sheet -> seven 28x28 MNIST-style digit crops.

Steps: grayscale -> find the 4 black corner markers (OpenCV contours) -> perspective warp to the
template size -> crop each box at its fixed position -> MNIST-style normalisation.
If the markers are not found we assume the photo is already cropped to the sheet and warn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evomarks.sheets import BOXES, CORNER_CENTERS, H, W

INNER_MARGIN = 14   # drops the printed box border from each crop


@dataclass
class PhotoResult:
    """crops: (7,28,28) uint8 digits; warped: aligned sheet; raw_boxes: unnormalised box crops."""
    crops: np.ndarray
    warped: np.ndarray
    raw_boxes: list = field(default_factory=list)
    found_markers: bool = True
    empty: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def find_corner_markers(gray: np.ndarray) -> np.ndarray | None:
    """Return the 4 marker centres ordered TL, TR, BL, BR, or None if not all found.

    The markers are the DARKEST things in the photo, so we try a ladder of strict dark thresholds
    (Otsu would merge a grey table with the markers) and keep the first that yields a big quadrilateral."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    for thr in (50, 70, 90, 110, 130):
        th = (blur < thr).astype(np.uint8) * 255
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # a marker is a small, solid, roughly square blob
            if 0.0003 * h * w < area < 0.02 * h * w and 0.6 < bw / bh < 1.6 and area / (bw * bh) > 0.7:
                cands.append((x + bw / 2, y + bh / 2))
        if len(cands) < 4:
            continue
        picks = [min(cands, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                 for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]]    # nearest candidate to each image corner
        quad = cv2.contourArea(cv2.convexHull(np.float32(picks)))
        if len(set(picks)) == 4 and quad > 0.2 * h * w:             # sheet must fill a good part of the photo
            return np.float32(picks)
    return None


def to_mnist(box: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalise one box crop (dark ink on light paper) to MNIST style. Returns (28x28, is_empty)."""
    blur = cv2.GaussianBlur(box, (3, 3), 0)
    if int(blur.max()) - int(blur.min()) < 50:
        return np.zeros((28, 28), np.uint8), True               # no ink at all
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    keep = np.isin(labels, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 25])   # drop specks
    if not keep.any():
        return np.zeros((28, 28), np.uint8), True
    ink = np.where(keep, 255 - blur, 0).astype(np.uint8)         # invert: white ink on black
    ys, xs = np.where(keep)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 20.0 / max(ink.shape)                                # longer side -> 20 px, like MNIST
    ink = cv2.resize(ink, (max(1, round(ink.shape[1] * scale)), max(1, round(ink.shape[0] * scale))),
                     interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.uint8)
    y0, x0 = (28 - ink.shape[0]) // 2, (28 - ink.shape[1]) // 2
    out[y0:y0 + ink.shape[0], x0:x0 + ink.shape[1]] = ink
    m = cv2.moments(out)                                         # centre by centre of mass
    shift = np.float32([[1, 0, 13.5 - m["m10"] / m["m00"]], [0, 1, 13.5 - m["m01"] / m["m00"]]])
    return cv2.warpAffine(out, shift, (28, 28)), False


def read_photo(source) -> PhotoResult:
    """Photo (path, PIL image or array) -> PhotoResult with 7 digit crops."""
    if isinstance(source, (str, Path)):
        source = Image.open(source)
    gray = np.array(source.convert("L")) if isinstance(source, Image.Image) else np.asarray(source)
    scale = 1600 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    warnings, corners = [], find_corner_markers(gray)
    if corners is None:
        warnings.append("Corner markers not found: assuming the photo is cropped to the sheet border.")
        warped = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    else:
        m = cv2.getPerspectiveTransform(corners, np.float32(CORNER_CENTERS))
        warped = cv2.warpPerspective(gray, m, (W, H), borderValue=255)
    crops, raw, empty = [], [], []
    for x, y, w, h in BOXES:
        box = warped[y + INNER_MARGIN:y + h - INNER_MARGIN, x + INNER_MARGIN:x + w - INNER_MARGIN]
        digit, is_empty = to_mnist(box)
        crops.append(digit), raw.append(box), empty.append(is_empty)
    if any(empty):
        warnings.append(f"{sum(empty)} box(es) look empty.")
    return PhotoResult(np.stack(crops), warped, raw, corners is not None, empty, warnings)
