"""The FIXED fitness function. The improver can never edit this file (it is protected).

`run()` evaluates one pipeline version: it launches a subprocess (timeout!) that loads the
version's pipeline.py, runs Level 0 under a fixed evolution budget and scores the final
teacher on dev (Level 1 fitness) and meta_dev (Level 2 utility). The model never grades itself:
scoring uses labeled splits that the candidate cannot reach (see safety.check).

Budgets: 'evo' = small, identical for every candidate; 'full' = the pipeline's own settings
(used once at the end for the best version).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import ROOT, Config, get_config
from evomarks import safety
from evomarks.data import load_split
from evomarks.metrics import count_params
from evomarks.self_training import run_level0
from evomarks.utils import load_json, load_module, save_json

TAMPER = {"checks": 0, "violations": 0}      # counted for the Safety page


def run(folder: Path, cfg: Config, seed: int = 0, budget: str = "evo", mode: str = "sum",
        out: str = "") -> dict:
    """Evaluate folder/pipeline.py in a sandboxed subprocess. Returns metrics or {'error': ...}."""
    folder = Path(folder)
    out_dir = folder / out if out else folder
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").unlink(missing_ok=True)
    before = safety.snapshot()
    cmd = [safety.PYTHON, "-m", "evomarks.evaluator", "--folder", str(folder), "--out", str(out_dir),
           "--seed", str(seed), "--budget", budget, "--mode", mode] + (["--quick"] if cfg.quick else [])
    code, _, err, timed_out = safety.run_sandboxed(cmd, cfg.eval_timeout_s * (6 if budget == "full" else 1))
    TAMPER["checks"] += 1
    try:
        safety.assert_unchanged(before)               # tamper check: fixed library/data/tests unchanged
    except safety.TamperError:
        TAMPER["violations"] += 1
        raise
    if timed_out:
        return {"error": "timeout"}
    if code != 0 or not (out_dir / "metrics.json").exists():
        return {"error": "crash", "detail": err.strip().splitlines()[-1] if err.strip() else ""}
    return load_json(out_dir / "metrics.json")


def worker(folder: Path, out_dir: Path, cfg: Config, seed: int, budget: str, mode: str) -> None:
    """Runs inside the subprocess: Level 0 with the candidate pipeline, then write metrics + model."""
    torch.set_num_threads(1)     # several evaluations run side by side; 1 thread each
    splits = {k: load_split(k) for k in ("seed", "pool", "dev", "meta_dev")}
    pipe = load_module(folder / "pipeline.py")
    if budget == "evo":
        rounds, per_round = cfg.evo_rounds, cfg.evo_pool_per_round
    else:
        rounds = pipe.SELF_TRAIN_ROUNDS
        per_round = min(pipe.POOL_PER_ROUND, cfg.n_pool // rounds)
    res = run_level0(pipe, splits, cfg, seed, rounds, per_round, mode, collect_examples=budget == "full")
    rl = res.rounds
    stats = {"sheets_read": sum(r["sheets_read"] for r in rl), "sheets_accepted": sum(r["accepted"] for r in rl),
             "digits_accepted": sum(r["accepted_digits"] for r in rl),
             "compensating_errors": sum(r["compensating_errors"] for r in rl),
             "low_conf_accepted": sum(r["low_conf_accepted"] for r in rl),
             "low_conf_correct": sum(r["low_conf_correct"] for r in rl)}
    metrics = {"dev_acc": res.final["acc"], "meta_acc": res.final["meta_acc"],
               "per_digit_acc": res.final["per_digit_acc"], "confusion": res.final["confusion"],
               "sheet_read_rate": res.final["exact_read_rate"], "baseline_dev_acc": res.baseline["acc"],
               "baseline_per_digit_acc": res.baseline["per_digit_acc"], "baseline_confusion": res.baseline["confusion"],
               "flag_precision": res.final["flag_precision"], "flag_recall": res.final["flag_recall"],
               "rounds": rl, "verifier_stats": stats, "train_time_s": res.train_time_s,
               "n_params": count_params(res.model), "budget": budget, "mode": mode, "seed": seed}
    torch.save(res.model.state_dict(), out_dir / "model.pt")
    if budget == "full":
        torch.save(res.seed_model.state_dict(), out_dir / "model_seed_only.pt")
        np.savez_compressed(out_dir / "l0_final.npz", **res.pseudo)
        _save_examples(res.examples, out_dir / "l0_examples.npz")
    save_json(metrics, out_dir / "metrics.json")     # written last: its presence means success


def _save_examples(ex: dict, path: Path) -> None:
    """Rescued low-confidence digits and rejected sheets, for figure l0_hard_examples."""
    r, j = ex.get("rescued", []), ex.get("rejected", [])
    np.savez_compressed(path, r_img=np.array([x[0] for x in r], dtype=np.uint8).reshape(-1, 28, 28),
                        r_pred=[x[1] for x in r], r_conf=[x[2] for x in r], r_true=[x[3] for x in r],
                        j_img=np.array([x[0] for x in j], dtype=np.uint8).reshape(-1, 7, 28, 28),
                        j_pred=np.array([x[1] for x in j]).reshape(-1, 7), j_true=np.array([x[2] for x in j]).reshape(-1, 7))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="evo")
    ap.add_argument("--mode", default="sum")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    worker(Path(a.folder), Path(a.out), get_config(True if a.quick else False), a.seed, a.budget, a.mode)
