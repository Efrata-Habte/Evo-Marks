"""Sum-check verifier: the heart of Level 0.

A model agreeing with itself is not proof. Arithmetic is.

The verifier only sees the *read* digits and the printed maxima. It never sees model
confidence or true labels, so the model cannot talk its way past it.

Known weakness: COMPENSATING ERRORS (two misreads that still sum correctly, e.g. a 3 read
as 5 and a 5 read as 3) pass the check. Genuinely miscounted sheets are correctly rejected
(the written total really is wrong) but cost us usable data. Both are measured
afterwards with the hidden labels (see self_training._audit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import MAXIMA


@dataclass
class VerificationResult:
    """accepted: sheet passes; reason: 'ok' | 'mark_exceeds_max' | 'sum_mismatch'."""
    accepted: bool
    reason: str


def verify(read_marks, read_total: int, maxima=MAXIMA) -> VerificationResult:
    """Accept only if every mark <= its maximum AND sum(marks) == total."""
    if any(m > mx for m, mx in zip(read_marks, maxima)):
        return VerificationResult(False, "mark_exceeds_max")
    if sum(read_marks) != read_total:
        return VerificationResult(False, "sum_mismatch")
    return VerificationResult(True, "ok")


def verify_batch(digits: np.ndarray, maxima=MAXIMA) -> tuple[np.ndarray, list[str]]:
    """Verify many sheets. digits: (N,7) read digits -> (accepted mask, reasons)."""
    reasons = []
    for d in digits:
        reasons.append(verify([int(x) for x in d[:5]], int(d[5]) * 10 + int(d[6]), maxima).reason)
    return np.array([r == "ok" for r in reasons]), reasons
