"""Model builders used by pipeline.py, plus the Reader wrapper (model + its preprocessing)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def small_cnn(channels: list[int], hidden: int, dropout: float) -> nn.Module:
    """One 3x3 conv (stride 2, so it also downsamples) + ReLU per entry of `channels`, then a hidden
    layer and 10 logits. Stride-2 convs instead of MaxPool: ~3x faster on plain CPUs, same accuracy class."""
    layers: list[nn.Module] = []
    c_in, size = 1, 28
    for c in channels:
        layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.ReLU()]
        c_in, size = c, (size + 1) // 2
    layers += [nn.Flatten(), nn.Linear(c_in * size * size, hidden), nn.ReLU(), nn.Dropout(dropout),
               nn.Linear(hidden, 10)]
    return nn.Sequential(*layers)


def mlp(hidden: int = 64, dropout: float = 0.0) -> nn.Module:
    """Plain MLP 784 -> hidden -> 10."""
    return nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 10))


def tiny_student(hidden: int = 32) -> nn.Module:
    """The distillation student: MLP 784 -> 32 -> 10 (~25k parameters, small enough for a phone)."""
    return mlp(hidden)


def predict_proba(model: nn.Module, imgs: np.ndarray, preprocess: Callable) -> np.ndarray:
    """Softmax probabilities (N,10) for raw uint8 digit images (N,28,28)."""
    x = torch.from_numpy(np.stack([preprocess(i) for i in imgs])[:, None].astype(np.float32))
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), 1).numpy()


@dataclass
class Reader:
    """A trained model together with the preprocessing it was trained with."""
    name: str
    model: nn.Module
    preprocess: Callable

    def probs(self, imgs: np.ndarray) -> np.ndarray:
        """Class probabilities for raw digit images."""
        return predict_proba(self.model, imgs, self.preprocess)
