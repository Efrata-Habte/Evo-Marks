"""MNIST loading and writer-disjoint split building.

WHY split this way: MNIST's official train and test sets were written by DIFFERENT people.
seed/pool/dev/meta_dev sheets are cut from MNIST *train*; test sheets from MNIST *test*,
so the final synthetic evaluation uses unseen writers. Each image is used in one split only.

  seed      (labeled)   -> teacher's first training
  pool      (unlabeled) -> Level 0 self-training (true labels only used for audits)
  dev                   -> Level 0 gate + Level 1 fitness
  meta_dev              -> Level 2 improver gate ONLY
  test                  -> reporting only (touched by 03_distill)
  real_sheets           -> final photographed sheets, run ONCE
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import ROOT, Config
from evomarks.sheets import DigitPool, SheetSet, generate_sheets
from evomarks.utils import save_json

GEN_DIR = ROOT / "data" / "generated"
SPLITS = ("seed", "pool", "dev", "meta_dev", "test")


def load_mnist(root: Path = ROOT / "data" / "mnist") -> dict:
    """Return {'train': (images, labels), 'test': (images, labels)} as numpy arrays."""
    from torchvision.datasets import MNIST
    out = {}
    for name, train in (("train", True), ("test", False)):
        ds = MNIST(str(root), train=train, download=True)
        out[name] = (ds.data.numpy(), ds.targets.numpy())
    return out


def build_splits(cfg: Config) -> dict[str, SheetSet]:
    """Generate all five synthetic splits deterministically from cfg.data_seed."""
    mn = load_mnist()
    rng = np.random.default_rng(cfg.data_seed)
    train_pool = DigitPool(mn["train"][1], rng)    # shared => train-derived splits never overlap
    test_pool = DigitPool(mn["test"][1], rng)
    sizes = {"seed": cfg.n_seed, "pool": cfg.n_pool, "dev": cfg.n_dev, "meta_dev": cfg.n_meta, "test": cfg.n_test}
    out = {}
    for name in SPLITS:
        src = "test" if name == "test" else "train"
        pool = test_pool if name == "test" else train_pool
        out[name] = generate_sheets(sizes[name], name, mn[src][0], pool, rng, cfg.miscount_rate)
    return out


def save_splits(splits: dict[str, SheetSet], cfg: Config, directory: Path = GEN_DIR) -> None:
    """Write each split as .npz plus meta.json (records quick mode and sizes)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        np.savez_compressed(directory / f"{name}.npz", images=s.images, true_marks=s.true_marks,
                            written_total=s.written_total, is_miscounted=s.is_miscounted, mnist_idx=s.mnist_idx)
    save_json({"quick": cfg.quick, "sizes": {k: len(v) for k, v in splits.items()}}, directory / "meta.json")


def load_split(name: str, directory: Path = GEN_DIR) -> SheetSet:
    """Load one split saved by save_splits."""
    z = np.load(directory / f"{name}.npz")
    return SheetSet(name, z["images"], z["true_marks"], z["written_total"], z["is_miscounted"], z["mnist_idx"])


def sheets_to_digits(s: SheetSet) -> tuple[np.ndarray, np.ndarray]:
    """Flatten sheets to digit arrays: images (N*7,28,28), labels (N*7). Labels are HIDDEN
    truth: only call this on labeled splits (seed) or when scoring (dev/meta/test)."""
    return s.images.reshape(-1, 28, 28), s.digits.reshape(-1)
