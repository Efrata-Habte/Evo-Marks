"""Digit / sheet metrics, confusion matrices, model size and CPU latency."""
from __future__ import annotations

import io
import time

import numpy as np
import torch

from evomarks.verifier import verify_batch


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """10x10 confusion matrix: rows = true digit, columns = predicted digit."""
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_digit_acc(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """Accuracy for each digit class 0..9 (nan-safe)."""
    return [float((y_pred[y_true == c] == c).mean()) if (y_true == c).any() else 1.0 for c in range(10)]


def sheet_metrics(pred: np.ndarray, digits: np.ndarray, miscounted: np.ndarray) -> dict:
    """Sheet-level scores. pred/digits: (N,7). A sheet is FLAGGED when the sum check fails.

    exact_read_rate: all 7 digits right. flag precision/recall are about the real-world job:
    flag sheets whose written total is genuinely wrong. False alarms come from misreads."""
    accepted, _ = verify_batch(pred)
    flagged = ~accepted
    exact = (pred == digits).all(axis=1)
    tp = int((flagged & miscounted).sum())
    return {"digit_acc": float((pred == digits).mean()), "exact_read_rate": float(exact.mean()),
            "flag_precision": tp / max(int(flagged.sum()), 1), "flag_recall": tp / max(int(miscounted.sum()), 1),
            "n_flagged": int(flagged.sum()), "false_alarms": int((flagged & ~miscounted).sum()),
            "missed": int((~flagged & miscounted).sum()),
            "false_alarms_from_misreads": int((flagged & ~miscounted & ~exact).sum())}


def count_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: torch.nn.Module) -> float:
    """Size of the float32 state_dict in kilobytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024


def cpu_latency_ms(reader, sheet_images: np.ndarray, runs: int = 200) -> float:
    """Median milliseconds to read ONE sheet (preprocess + forward of 7 digits), 1 CPU thread.

    Includes warm-up runs; torch threads pinned to 1 so numbers are comparable to a phone core."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(20):
            reader.probs(sheet_images[0])
        times = []
        for i in range(runs):
            s = sheet_images[i % len(sheet_images)]
            t0 = time.perf_counter()
            reader.probs(s)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(prev)
    return float(np.median(times))
