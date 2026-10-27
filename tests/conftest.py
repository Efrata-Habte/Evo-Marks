"""Shared fixtures: tiny synthetic splits built from the real MNIST files."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from evomarks.data import build_splits


@pytest.fixture(scope="session")
def tiny_splits():
    """Small deterministic splits (a few seconds to build)."""
    cfg = Config(n_seed=20, n_pool=100, n_dev=20, n_meta=20, n_test=40, miscount_rate=0.2)
    return build_splits(cfg)
