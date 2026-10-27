"""Knowledge distillation: a big teacher's soft targets train a tiny student.

L = alpha * CE(y, student) + (1 - alpha) * T^2 * KL( softmax(teacher/T) || softmax(student/T) )

* T > 1 softens the teacher's distribution, exposing "dark knowledge" (a 4 that looks a bit like a 9).
* T^2 restores the gradient scale that the 1/T^2 softening removes.
* alpha = 1 means NO distillation: pure hard-label cross-entropy (the baseline student).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor,
                      T: float, alpha: float) -> torch.Tensor:
    """The slide formula. alpha=1 reduces exactly to cross-entropy."""
    ce = F.cross_entropy(student_logits, y)
    kl = F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1),
                  reduction="batchmean") * T * T
    return alpha * ce + (1 - alpha) * kl


def teacher_logits_for(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Teacher outputs, computed in eval() mode under no_grad (the teacher is never updated)."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + 2048]) for i in range(0, len(x), 2048)])


def train_student(student: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                  t_logits: torch.Tensor | None, T: float, alpha: float,
                  epochs: int, seed: int, lr: float = 3e-3, batch: int = 128) -> torch.nn.Module:
    """Train `student` on inputs x (N,1,28,28) with labels y. t_logits=None -> hard-label baseline."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    student.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen)
        for b in range(0, len(y), batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            out = student(x[idx])
            loss = F.cross_entropy(out, y[idx]) if t_logits is None else \
                distillation_loss(out, t_logits[idx], y[idx], T, alpha)
            loss.backward()
            opt.step()
    return student.eval()


def to_input(images: np.ndarray) -> torch.Tensor:
    """Raw uint8 digits (N,28,28) -> student input tensor (N,1,28,28) in [0,1] (normalize only)."""
    return torch.from_numpy(images.astype(np.float32) / 255.0)[:, None]
