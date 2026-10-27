"""ALL settings in one place: seeds, split sizes, budgets, gates, KD grid, quick mode."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAXIMA = (5, 8, 9, 6, 7)   # printed maximum mark of Q1..Q5
N_QUESTIONS = 5
N_BOXES = 7                # 5 marks + 2 total digits


@dataclass(frozen=True)
class Config:
    """Every knob of the project. `Config.make(quick)` scales sizes down for the demo."""
    quick: bool = False
    data_seed: int = 123            # fixes the synthetic sheets (reproducibility)
    seed: int = 0                   # model / evolution seed
    miscount_rate: float = 0.05     # share of sheets whose written total is wrong
    # --- split sizes (in sheets of 7 digits) ---
    n_seed: int = 150               # labeled
    n_pool: int = 3000              # unlabeled (hidden labels, audit only)
    n_dev: int = 300                # Level 0 gate + Level 1 fitness
    n_meta: int = 300               # Level 2 improver gate only
    n_test: int = 750               # reporting only (MNIST *test* writers; 1000 would need more 1s than MNIST-test has)
    # --- Level 0 gate ---
    max_class_drop: float = 0.05    # reject if any digit class loses > 5 points on dev
    conf_threshold: float = 0.9     # ablation: confidence-threshold pseudo-labels
    low_conf: float = 0.6           # "hard example" = read with confidence below this
    patience: int = 2               # stop after this many consecutive rejections
    # --- evolution budget (fixed for every candidate: fair evaluator) ---
    evo_rounds: int = 2
    evo_pool_per_round: int = 400
    generations: int = 20           # G
    improver_every: int = 5         # K
    utility_edits: int = 3          # B edits per improver when measuring utility
    min_delta: float = 0.002        # Level 1 acceptance margin
    min_delta_l2: float = 0.001     # Level 2 acceptance margin (utility)
    eval_timeout_s: int = 150
    evo_seeds: int = 3              # independent evolution runs per variant (L1 only vs L1+L2); quick mode uses 1
    # --- distillation ---
    kd_seeds: tuple = (0, 1, 2)
    temperatures: tuple = (1.0, 2.0, 4.0)
    alphas: tuple = (0.3, 0.7)
    student_hidden: int = 32
    student_epochs: int = 15
    latency_runs: int = 200

    @classmethod
    def make(cls, quick: bool = False) -> "Config":
        """Full config, or the ~5 minute demo config."""
        if not quick:
            return cls()
        return cls(quick=True, n_seed=60, n_pool=800, n_dev=120, n_meta=120, n_test=300,
                   evo_rounds=2, evo_pool_per_round=150, generations=6, improver_every=3,
                   utility_edits=2, eval_timeout_s=90, evo_seeds=1, student_epochs=8, latency_runs=200)


def get_config(quick: bool | None = None) -> Config:
    """quick=None -> reuse the mode recorded when the sheets were generated."""
    if quick is None:
        meta = ROOT / "data" / "generated" / "meta.json"
        quick = bool(json.loads(meta.read_text())["quick"]) if meta.exists() else \
            os.environ.get("EVOMARKS_QUICK") == "1"
    return Config.make(quick)
