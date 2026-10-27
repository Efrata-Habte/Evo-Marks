"""Distillation loss and the photo reader under perspective."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evomarks.distillation import distillation_loss
from evomarks.photo import read_photo
from evomarks.sheets import render_sheet


def test_alpha_one_equals_cross_entropy():
    s, t = torch.randn(32, 10), torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    for T in (1.0, 2.0, 4.0):
        assert torch.allclose(distillation_loss(s, t, y, T, alpha=1.0), F.cross_entropy(s, y), atol=1e-6)


def test_alpha_zero_is_scaled_kl_and_zero_for_identical_logits():
    s = torch.randn(16, 10)
    y = torch.randint(0, 10, (16,))
    assert distillation_loss(s, s.clone(), y, T=4.0, alpha=0.0).abs() < 1e-6   # student == teacher -> no KL


def test_read_photo_survives_perspective_and_grey_background(tiny_splits):
    img = np.array(render_sheet(tiny_splits["dev"], 1).convert("L"))
    h, w = img.shape
    bg = np.full((h + 400, w + 400), 120, np.uint8)
    bg[200:200 + h, 200:200 + w] = img
    src = np.float32([[200, 200], [200 + w, 200], [200, 200 + h], [200 + w, 200 + h]])
    dst = np.float32([[260, 230], [200 + w - 60, 180], [220, 200 + h + 40], [200 + w + 30, 200 + h - 30]])
    photo = cv2.warpPerspective(bg, cv2.getPerspectiveTransform(src, dst), (w + 400, h + 400), borderValue=120)
    res = read_photo(photo)
    assert res.found_markers and res.crops.shape == (7, 28, 28)
    assert res.crops.reshape(7, -1).max(axis=1).min() > 100          # every box contains ink
