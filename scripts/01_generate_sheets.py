"""Step 1: generate the synthetic sheets + splits from MNIST (deterministic)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config
from evomarks.data import build_splits, save_splits
from evomarks.utils import set_seed


def main() -> None:
    """Build, sanity-check and save the five splits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    cfg = get_config(ap.parse_args().quick)
    set_seed(cfg.seed)
    splits = build_splits(cfg)
    # Evidence for the report: no MNIST train image may sit in two splits
    seen: set = set()
    for k, s in splits.items():
        if k == "test":
            continue
        ids = list(map(int, s.mnist_idx.ravel()))
        assert not (seen & set(ids)) and len(set(ids)) == len(ids), f"image reused in {k}"
        seen |= set(ids)
    save_splits(splits, cfg)
    for k, s in splits.items():
        print(f"{k:9s} {len(s):5d} sheets  miscounted={int(s.is_miscounted.sum()):4d}")
    print("quick mode" if cfg.quick else "full mode", "- saved to data/generated/")


if __name__ == "__main__":
    main()
