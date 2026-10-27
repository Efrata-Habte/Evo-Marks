"""Whitelisted preprocessing / augmentation building blocks.

`evolvable/pipeline.py` may only compose these. Every block takes a 28x28 image (uint8 or
float) and returns a float32 28x28 image in [0, 1]. Augmentation blocks also take a numpy
Generator `rng` so runs stay reproducible. Small, readable cv2/numpy code on purpose.
"""
from __future__ import annotations

import cv2
import numpy as np

cv2.setNumThreads(1)
_K = np.ones((2, 2), np.uint8)


def _f(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]; floats pass through."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _affine(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix with black borders."""
    return cv2.warpAffine(_f(img), m, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)


# ---------------- preprocessing ----------------
def normalize(img: np.ndarray) -> np.ndarray:
    """Scale to float [0,1]. Version 0 uses only this."""
    return _f(img)


def center_by_mass(img: np.ndarray) -> np.ndarray:
    """Shift the digit so its centre of mass sits at the image centre."""
    img = _f(img)
    m = cv2.moments(img)
    if m["m00"] < 1e-3:
        return img
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return _affine(img, np.float32([[1, 0, 13.5 - cx], [0, 1, 13.5 - cy]]))


def deskew(img: np.ndarray) -> np.ndarray:
    """Undo slant using second-order image moments."""
    img = _f(img)
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:
        return img
    skew = m["mu11"] / m["mu02"]
    return _affine(img, np.float32([[1, skew, -0.5 * 28 * skew], [0, 1, 0]]))


def dilate(img: np.ndarray) -> np.ndarray:
    """Thicken strokes."""
    return cv2.dilate(_f(img), _K)


def erode(img: np.ndarray) -> np.ndarray:
    """Thin strokes."""
    return cv2.erode(_f(img), _K)


def gaussian_blur(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Smooth pixel noise."""
    return cv2.GaussianBlur(_f(img), (0, 0), sigma)


def clip_contrast(img: np.ndarray, low: float = 0.1) -> np.ndarray:
    """Zero out faint background below `low`, stretch the rest back to [0,1]."""
    return np.clip((_f(img) - low) / (1.0 - low), 0.0, 1.0)


# ---------------- augmentation (training only) ----------------
def random_rotate(img: np.ndarray, rng: np.random.Generator, deg: float = 10) -> np.ndarray:
    """Rotate by a random angle in [-deg, deg]."""
    if deg <= 0:
        return _f(img)
    return _affine(img, cv2.getRotationMatrix2D((13.5, 13.5), float(rng.uniform(-deg, deg)), 1.0))


def random_shift(img: np.ndarray, rng: np.random.Generator, px: int = 2) -> np.ndarray:
    """Translate by up to `px` pixels in x and y."""
    if px <= 0:
        return _f(img)
    dx, dy = rng.integers(-px, px + 1, size=2)
    return _affine(img, np.float32([[1, 0, dx], [0, 1, dy]]))


def random_elastic(img: np.ndarray, rng: np.random.Generator, alpha: float = 2.0) -> np.ndarray:
    """Smooth random warp (mimics handwriting wobble)."""
    img = _f(img)
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (28, 28)).astype(np.float32), (0, 0), 3) * alpha * 4
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (28, 28)).astype(np.float32), (0, 0), 3) * alpha * 4
    gx, gy = np.meshgrid(np.arange(28, dtype=np.float32), np.arange(28, dtype=np.float32))
    return cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)


def random_noise(img: np.ndarray, rng: np.random.Generator, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian pixel noise."""
    return np.clip(_f(img) + rng.normal(0, sigma, (28, 28)).astype(np.float32), 0.0, 1.0)
