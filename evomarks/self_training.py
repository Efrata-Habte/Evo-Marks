"""Level 0: verified self-training (changes weights, not code).

The teacher reads unlabeled sheets; a sheet's digits become pseudo-labels ONLY if the read marks
sum to the read total and every mark is within its maximum. Accepted sheets train a COPY of the
teacher, which replaces it only if it passes the dev gate; otherwise rollback.
All settings (epochs, lr, augmentation...) come from the pipeline module being evaluated.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from evomarks.data import sheets_to_digits
from evomarks.metrics import confusion, per_digit_acc, sheet_metrics
from evomarks.models import predict_proba
from evomarks.utils import set_seed
from evomarks.verifier import verify_batch


@dataclass
class Level0Result:
    """Everything the evaluator / figures need from one Level 0 run."""
    model: torch.nn.Module
    seed_model: torch.nn.Module                 # teacher after seed-only training = "no loops" baseline
    baseline: dict                              # dev metrics of the seed-only teacher
    final: dict                                 # dev metrics of the final teacher (+ meta_acc)
    rounds: list = field(default_factory=list)
    pseudo: dict = None                         # kept pseudo-labels: sheet_idx, box, labels (arrays)
    examples: dict = field(default_factory=dict)
    train_time_s: float = 0.0


def train(model, imgs: np.ndarray, labels: np.ndarray, pipe, seed: int) -> None:
    """Train in place for pipe.EPOCHS. Augmentation is re-drawn every epoch."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=pipe.LR)
    y = torch.from_numpy(labels.astype(np.int64))
    model.train()
    for _ in range(pipe.EPOCHS):
        x = torch.from_numpy(np.stack([pipe.preprocess(pipe.augment(i, rng)) for i in imgs])[:, None].astype(np.float32))
        perm = torch.from_numpy(rng.permutation(len(y)))
        for b in range(0, len(y), pipe.BATCH):
            idx = perm[b:b + pipe.BATCH]
            opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()


def evaluate(model, pipe, sheets) -> dict:
    """Score a model on a labeled split (dev / meta_dev). Labels are used for scoring only."""
    imgs, y = sheets_to_digits(sheets)
    pred = predict_proba(model, imgs, pipe.preprocess).argmax(1)
    out = {"acc": float((pred == y).mean()), "per_digit_acc": per_digit_acc(y, pred),
           "confusion": confusion(y, pred).tolist()}
    out.update(sheet_metrics(pred.reshape(-1, 7), sheets.digits, sheets.is_miscounted))
    return out


def select_digits(probs: np.ndarray, mode: str, cfg: Config) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Which digits become pseudo-labels? probs: (N,7,10) -> (digit mask (N,7), sheet ok (N,), reasons).

    mode 'sum': the arithmetic verifier accepts or rejects WHOLE sheets (the method).
    mode 'confidence': each digit is kept if its own confidence >= threshold, no sheet structure used
    (the standard self-training baseline = a model agreeing with itself)."""
    if mode == "sum":
        ok, reasons = verify_batch(probs.argmax(2))
        return np.repeat(ok[:, None], 7, axis=1), ok, reasons
    mask = probs.max(2) >= cfg.conf_threshold
    ok = mask.any(1)
    return mask, ok, ["ok" if o else "low_confidence" for o in ok]


def _audit(probs, mask, ok, true_digits, cfg) -> dict:
    """AUDIT ONLY, runs after the decision and never feeds back: uses the hidden labels to measure
    pseudo-label precision, compensating errors and the low-confidence digits that were rescued."""
    pred, conf = probs.argmax(2), probs.max(2)
    right = pred == true_digits
    low = (conf < cfg.low_conf) & mask
    return {"accepted_digits": int(mask.sum()),
            "verifier_precision": float(right[mask].mean()) if mask.any() else None,
            "compensating_errors": int((~right.all(1) & mask.all(1)).sum()),    # accepted sheet, wrong digits, sums fine
            "low_conf_accepted": int(low.sum()), "low_conf_correct": int((right & low).sum()),
            "rejected_genuine_miscount": int((right.all(1) & ~ok).sum()),       # read perfectly, total really wrong
            "rejected_misread": int((~right.all(1) & ~ok).sum())}


def run_level0(pipe, splits: dict, cfg: Config, seed: int, rounds: int, pool_per_round: int,
               mode: str = "sum", collect_examples: bool = False) -> Level0Result:
    """Seed-train a teacher, then run `rounds` verified self-training rounds with rollback."""
    t0 = time.time()
    set_seed(seed)
    seed_imgs, seed_y = sheets_to_digits(splits["seed"])
    pool, dev = splits["pool"], splits["dev"]
    teacher = pipe.build_teacher()
    train(teacher, seed_imgs, seed_y, pipe, seed)
    seed_model = copy.deepcopy(teacher)
    baseline = evaluate(teacher, pipe, dev)
    best, cur = baseline["acc"], baseline
    order = np.random.default_rng(seed).permutation(len(pool))      # pool chunks never repeat
    keep = {k: np.zeros(0, dtype=int) for k in ("sheet_idx", "box", "labels")}   # committed pseudo-labels
    logs, examples, misses = [], {"rescued": [], "rejected": []}, 0
    for r in range(1, rounds + 1):
        idx = order[(r - 1) * pool_per_round: r * pool_per_round]
        if len(idx) == 0:
            break
        imgs = pool.images[idx]                                      # ONLY images: labels are hidden
        probs = predict_proba(teacher, imgs.reshape(-1, 28, 28), pipe.preprocess).reshape(-1, 7, 10)
        mask, ok_sheets, reasons = select_digits(probs, mode, cfg)
        audit = _audit(probs, mask, ok_sheets, pool.digits[idx], cfg)       # hidden labels: audit only
        log = {"round": r, "sheets_read": len(idx), "accepted": int(ok_sheets.sum()),
               "rejected": {k: reasons.count(k) for k in set(reasons) if k != "ok"}, **audit}
        if collect_examples:
            _collect(examples, probs, mask, reasons, imgs, pool.digits[idx], cfg)
        cand = copy.deepcopy(teacher)                                # train a COPY, never the teacher itself
        sh, bx = np.where(mask)
        new = {"sheet_idx": np.concatenate([keep["sheet_idx"], idx[sh]]), "box": np.concatenate([keep["box"], bx]),
               "labels": np.concatenate([keep["labels"], probs.argmax(2)[sh, bx]])}
        if mask.any():
            train(cand, np.concatenate([seed_imgs, pool.images[new["sheet_idx"], new["box"]]]),
                  np.concatenate([seed_y, new["labels"]]), pipe, seed * 100 + r)
        cand_m = evaluate(cand, pipe, dev)
        drop = max(a - b for a, b in zip(cur["per_digit_acc"], cand_m["per_digit_acc"]))
        # Gate: reject if dev accuracy does not improve -> guards against self-confirming errors
        improved = cand_m["acc"] >= best + pipe.MIN_DELTA
        ok = bool(mask.any()) and improved and drop <= cfg.max_class_drop
        reason = "ok" if ok else ("nothing_accepted" if not mask.any() else "class_drop" if improved else "no_improvement")
        log.update(decision="accept" if ok else "reject", reason=reason, candidate_dev_acc=cand_m["acc"],
                   per_digit_acc=cand_m["per_digit_acc"], n_train=len(seed_y) + len(new["labels"]))
        if ok:
            teacher, best, cur, keep, misses = cand, cand_m["acc"], cand_m, new, 0
        else:                                                        # rollback: teacher and data unchanged
            misses += 1
        log["dev_acc"] = best
        logs.append(log)
        if misses >= cfg.patience:
            break
    final = dict(cur, meta_acc=evaluate(teacher, pipe, splits["meta_dev"])["acc"])
    return Level0Result(teacher, seed_model, baseline, final, logs, keep, examples, time.time() - t0)


def _collect(examples: dict, probs, mask, reasons, imgs, truth, cfg: Config, cap: int = 48) -> None:
    """Keep the FIRST few rescued low-confidence digits and rejected sheets (no cherry-picking)."""
    pred, conf = probs.argmax(2), probs.max(2)
    for i, j in zip(*np.where(mask & (conf < cfg.low_conf))):
        if len(examples["rescued"]) < cap:
            examples["rescued"].append((imgs[i, j], int(pred[i, j]), float(conf[i, j]), int(truth[i, j])))
    for i, rs in enumerate(reasons):
        if rs == "sum_mismatch" and len(examples["rejected"]) < 16:
            examples["rejected"].append((imgs[i], pred[i].copy(), truth[i].copy()))
