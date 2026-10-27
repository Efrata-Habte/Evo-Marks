"""AST-level source editing: constrained program search (no LLM, no API).

Each operator parses the source with `ast`, modifies the tree, and `ast.unparse`s it back, so
every candidate is syntactically valid by construction. This is a small-scale, rule-based version
of what the Darwin Goedel Machine does with an LLM: propose a code edit, then let a FIXED
evaluator decide. Operators return (new_source, human-readable description).
Note: ast.unparse drops comments; docstrings survive.
"""
from __future__ import annotations

import ast
import math
import random


class NoEdit(Exception):
    """Raised when an operator cannot change the source (e.g. nothing to remove)."""


# name -> (low, high, type). Bounds keep every candidate affordable and sane.
PIPE_BOUNDS = {"HIDDEN": (16, 128, int), "DROPOUT": (0.0, 0.5, float), "LR": (3e-4, 1e-2, float),
               "EPOCHS": (2, 14, int), "BATCH": (32, 256, int), "ROT_DEG": (0, 25, int), "SHIFT_PX": (0, 3, int)}
AUG_CONSTS = {"ROT_DEG": "random_rotate", "SHIFT_PX": "random_shift"}
BLOCKS = {  # function -> block -> statement template
    "preprocess": {"center_by_mass": "img = blocks.center_by_mass(img)", "deskew": "img = blocks.deskew(img)",
                   "dilate": "img = blocks.dilate(img)", "erode": "img = blocks.erode(img)",
                   "gaussian_blur": "img = blocks.gaussian_blur(img, 0.6)",
                   "clip_contrast": "img = blocks.clip_contrast(img, 0.1)"},
    "augment": {"random_rotate": "img = blocks.random_rotate(img, rng, ROT_DEG)",
                "random_shift": "img = blocks.random_shift(img, rng, SHIFT_PX)",
                "random_elastic": "img = blocks.random_elastic(img, rng, 1.5)",
                "random_noise": "img = blocks.random_noise(img, rng, 0.05)"}}


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    """Find a top-level function."""
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise NoEdit(f"no function {name}")


def _assign(tree: ast.Module, name: str) -> ast.Assign:
    """Find a top-level `NAME = ...` statement."""
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return n
    raise NoEdit(f"no constant {name}")


def _block_name(stmt: ast.stmt) -> str | None:
    """`img = blocks.X(...)` -> 'X', otherwise None."""
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute):
        return stmt.value.func.attr
    return None


def _block_calls(tree: ast.Module, fn: str) -> list[str]:
    """Names of the blocks currently used in a function."""
    return [b for b in map(_block_name, _func(tree, fn).body) if b]


def _step(value, lo, hi, kind, rng: random.Random, scale: float):
    """Move a constant up or down by ~scale (relative), respecting type and bounds."""
    if kind is int:
        delta = max(1, round(abs(value) * scale * rng.uniform(0.5, 1.5)))
        new = value + rng.choice([-1, 1]) * delta
        return int(min(max(new, lo), hi))
    factor = math.exp(rng.choice([-1, 1]) * scale * rng.uniform(0.5, 1.5))
    return float(min(max(round(value * factor, 5), lo), hi))


def _finish(tree: ast.Module, old_src: str) -> str:
    """Unparse and make sure something really changed."""
    new = ast.unparse(ast.fix_missing_locations(tree))
    if new == ast.unparse(ast.parse(old_src)):
        raise NoEdit("edit produced identical source")
    return new + "\n"


def tweak_constant(src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Change one numeric training constant (LR, EPOCHS, DROPOUT, ...) within bounds."""
    tree = ast.parse(src)
    aug_used = _block_calls(tree, "augment")
    names = [n for n in PIPE_BOUNDS if n not in AUG_CONSTS or AUG_CONSTS[n] in aug_used]   # tune augmentation only if used
    name = rng.choice(names)
    node = _assign(tree, name)
    lo, hi, kind = PIPE_BOUNDS[name]
    old = node.value.value
    node.value = ast.Constant(_step(old, lo, hi, kind, rng, scale))
    return _finish(tree, src), f"tweak_constant: {name} {old} -> {node.value.value}"


def add_block(src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Insert one whitelisted block call into preprocess or augment."""
    tree = ast.parse(src)
    fn = rng.choice(list(BLOCKS))
    free = [b for b in BLOCKS[fn] if b not in _block_calls(tree, fn)]
    if not free:
        raise NoEdit("all blocks already used")
    block = rng.choice(free)
    body = _func(tree, fn).body
    lo = 2 if fn == "preprocess" else 1                       # keep the docstring (and `normalize`) first
    pos = rng.randint(lo, max(lo, len(body) - 1))             # always before the final return
    body.insert(pos, ast.parse(BLOCKS[fn][block]).body[0])
    extra = ""
    for const, blk in AUG_CONSTS.items():                     # a 0-strength rotate/shift would do nothing
        if blk == block and _assign(tree, const).value.value == 0:
            _assign(tree, const).value = ast.Constant(8 if const == "ROT_DEG" else 2)
            extra = f" ({const}={_assign(tree, const).value.value})"
    return _finish(tree, src), f"add_block: {fn} += {block}{extra}"


def remove_block(src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Delete one block call (never `normalize`, never the return)."""
    tree = ast.parse(src)
    cands = [(fn, s) for fn in BLOCKS for s in _func(tree, fn).body
             if _block_name(s) and _block_name(s) != "normalize"]
    if not cands:
        raise NoEdit("no removable block")
    fn, stmt = rng.choice(cands)
    _func(tree, fn).body.remove(stmt)
    return _finish(tree, src), f"remove_block: {fn} -= {_block_name(stmt)}"


def resize_model(src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Change CONV_CHANNELS (widths or depth) or HIDDEN."""
    tree = ast.parse(src)
    node = _assign(tree, "CONV_CHANNELS")
    old = [e.value for e in node.value.elts]
    new = list(old)
    r = rng.random()
    if r < 0.2 and len(new) < 3:
        new.append(max(4, min(32, new[-1] * 2)))              # deeper
    elif r < 0.3 and len(new) > 1:
        new.pop()                                             # shallower
    else:
        i = rng.randrange(len(new))
        new[i] = int(min(max(round(new[i] * math.exp(rng.choice([-1, 1]) * scale * 1.5)), 4), 32))
    if new == old:
        return tweak_hidden(src, rng, scale)
    node.value = ast.List([ast.Constant(v) for v in new], ast.Load())
    return _finish(tree, src), f"resize_model: CONV_CHANNELS {old} -> {new}"


def tweak_hidden(src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Change the HIDDEN width of the classifier head."""
    tree = ast.parse(src)
    node = _assign(tree, "HIDDEN")
    old = node.value.value
    node.value = ast.Constant(_step(old, 16, 128, int, rng, scale * 1.5))
    return _finish(tree, src), f"resize_model: HIDDEN {old} -> {node.value.value}"


OPERATORS = {"tweak_constant": tweak_constant, "add_block": add_block,
             "remove_block": remove_block, "resize_model": resize_model}


def apply_operator(name: str, src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Run an operator; if it cannot edit, fall back to tweak_constant (then to 'no edit')."""
    for op in (OPERATORS[name], tweak_constant):
        for _ in range(4):
            try:
                return op(src, rng, scale)
            except NoEdit:
                continue
    return src, "no_edit"


# ---------------- edits of the improver's OWN source (Level 2) ----------------
IMPROVER_BOUNDS = {"EDITS_PER_CHILD": (1, 3, int), "TOURNAMENT_K": (2, 6, int), "STEP_SCALE": (0.1, 1.0, float)}
SELECTIONS = ["tournament", "best", "recent"]


def tweak_improver_constant(src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Change an operator weight or one of the improver's own search settings."""
    tree = ast.parse(src)
    weights = _assign(tree, "OPERATOR_WEIGHTS").value
    keys = [k.value for k in weights.keys]
    pick = rng.choice(keys + list(IMPROVER_BOUNDS))
    if pick in keys:
        i = keys.index(pick)
        old = weights.values[i].value
        weights.values[i] = ast.Constant(_step(old, 0.05, 3.0, float, rng, 0.6))
        desc = f"OPERATOR_WEIGHTS['{pick}'] {old} -> {weights.values[i].value}"
    else:
        node, (lo, hi, kind) = _assign(tree, pick), IMPROVER_BOUNDS[pick]
        old = node.value.value
        node.value = ast.Constant(_step(old, lo, hi, kind, rng, 0.5 if kind is float else 0.4))
        desc = f"{pick} {old} -> {node.value.value}"
    return _finish(tree, src), "improver self-edit: " + desc


def swap_parent_selection(src: str, rng: random.Random, scale: float) -> tuple[str, str]:
    """Switch how the improver picks parents (tournament / best / recent)."""
    tree = ast.parse(src)
    node = _assign(tree, "PARENT_SELECTION")
    old = node.value.value
    node.value = ast.Constant(rng.choice([s for s in SELECTIONS if s != old]))
    return _finish(tree, src), f"improver self-edit: PARENT_SELECTION {old} -> {node.value.value}"


def edit_improver(src: str, rng: random.Random) -> tuple[str, str]:
    """One random self-edit of improver.py (used by improver.propose_self_edit)."""
    op = rng.choice([tweak_improver_constant, tweak_improver_constant, swap_parent_selection])
    for _ in range(5):
        try:
            return op(src, rng, 0.3)
        except NoEdit:
            continue
    return src, "no_edit"
