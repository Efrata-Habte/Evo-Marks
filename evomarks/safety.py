"""Guard-rails: what an evolved file may do, and proof that the fixed parts were not touched.

1. Static check (`check`): candidate source may only import a whitelist and may not mention
   dangerous names (open/exec/eval/os/...) or anything that touches the verifier, labels or splits.
2. Tamper detection (`snapshot` / `assert_unchanged`): SHA-256 of the fixed library, the generated
   sheets and the tests before and after every evaluation; abort on mismatch.
3. Sandbox (`run_sandboxed`): candidates run in a subprocess with a timeout.
Honest limit: this is a static whitelist + process isolation, not an OS-level sandbox.
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from config import ROOT

ALLOWED_IMPORTS = {"torch", "torch.nn", "numpy", "math", "random", "evomarks.blocks", "evomarks.models"}
IMPROVER_IMPORTS = {"math", "random", "evomarks.code_editor"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "globals",
                   "locals", "vars", "input", "os", "sys", "subprocess", "importlib", "shutil", "pathlib"}
FORBIDDEN_TOKENS = {"verifier", "data", "labels", "label", "split", "test", "evaluator", "safety"}
REQUIRED = {"pipeline": ["preprocess", "augment", "build_teacher", "CONV_CHANNELS", "HIDDEN", "DROPOUT", "LR",
                         "EPOCHS", "BATCH", "SELF_TRAIN_ROUNDS", "POOL_PER_ROUND", "MIN_DELTA"],
            "improver": ["choose_parent", "propose_child", "propose_self_edit", "OPERATOR_WEIGHTS"]}
PROTECTED = [ROOT / "evomarks", ROOT / "data" / "generated", ROOT / "tests", ROOT / "config.py"]
EDITABLE_ROOTS = [ROOT / "archive", ROOT / "results", ROOT / "evolvable"]


@dataclass
class SafetyResult:
    """ok / reason ('ok' or e.g. 'forbidden_import:os')."""
    ok: bool
    reason: str


class TamperError(RuntimeError):
    """A protected file changed during an evaluation."""


def check(source: str, kind: str = "pipeline") -> SafetyResult:
    """Static safety check of a candidate pipeline.py / improver.py source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return SafetyResult(False, f"syntax_error:{e.msg}")
    allowed = ALLOWED_IMPORTS if kind == "pipeline" else IMPROVER_IMPORTS
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name not in allowed]
            if bad:
                return SafetyResult(False, f"forbidden_import:{bad[0]}")
        elif isinstance(node, (ast.While, ast.AsyncFunctionDef, ast.Global)):
            return SafetyResult(False, f"forbidden_construct:{type(node).__name__}")   # no unbounded loops
        elif isinstance(node, ast.ImportFrom):
            full = [f"{node.module}.{a.name}" for a in node.names]
            if node.module not in allowed and not all(f in allowed for f in full):
                return SafetyResult(False, f"forbidden_import:{node.module}")
        names = [getattr(node, f) for f in ("id", "attr", "arg", "name") if isinstance(getattr(node, f, None), str)]
        for n in names:
            if n in FORBIDDEN_NAMES or n.startswith("__"):
                return SafetyResult(False, f"forbidden_name:{n}")
            if FORBIDDEN_TOKENS & set(n.lower().split("_")):
                return SafetyResult(False, f"forbidden_reference:{n}")
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            defined.update([node.name] if isinstance(node, ast.FunctionDef) else
                           [t.id for t in node.targets if isinstance(t, ast.Name)])
    missing = [r for r in REQUIRED[kind] if r not in defined]
    return SafetyResult(not missing, f"missing:{missing[0]}" if missing else "ok")


def _hash_tree(path: Path) -> str:
    """SHA-256 over all .py/.npz/.json files under path (ignores caches)."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()
                                                  and p.suffix in {".py", ".npz", ".json"} and "__pycache__" not in p.parts)
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def snapshot(paths: list[Path] | None = None) -> dict[str, str]:
    """Hash every protected path (missing paths hash as empty)."""
    return {str(p.relative_to(ROOT)): _hash_tree(p) if p.exists() else "" for p in (paths or PROTECTED)}


def assert_unchanged(before: dict[str, str], paths: list[Path] | None = None) -> None:
    """Raise TamperError if any protected path differs from the earlier snapshot."""
    after = snapshot(paths)
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise TamperError(f"protected files changed: {changed}")


def run_sandboxed(cmd: list[str], timeout: int) -> tuple[int, str, str, bool]:
    """Run a subprocess with a timeout. Returns (returncode, stdout, stderr, timed_out)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONHASHSEED": "0"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT), env=env)
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired:
        return -1, "", "timeout", True


PYTHON = sys.executable
