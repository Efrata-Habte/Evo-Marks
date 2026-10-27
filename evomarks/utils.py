"""Small helpers shared everywhere: seeding, JSON, dynamic module loading."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so every run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _Encoder(json.JSONEncoder):
    """Lets json.dump handle numpy scalars/arrays and Paths."""

    def default(self, o):  # noqa: D102
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path: Path) -> None:
    """Write obj as pretty JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, cls=_Encoder))


def load_json(path: Path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


def now_iso() -> str:
    """Current local time as an ISO string (for logs)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_module(path: Path, name: str = "candidate") -> ModuleType:
    """Import a python file by path (used to load archived pipeline/improver versions)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # keep archive folders free of __pycache__
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod
