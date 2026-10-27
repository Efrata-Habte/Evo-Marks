"""LEVEL 2 TARGET: the improver, which proposes edits to pipeline.py AND to this very file.

`propose_child` edits a pipeline; `propose_self_edit` edits THIS source (operator weights, step
size, parent-selection rule). A new improver is kept only if, with the same budget, it improves
the pipeline MORE than the old one (STOP, Zelikman et al. 2023), measured on meta_dev.
It cannot touch the evaluator, the verifier or any data: safety.check forbids those references.
"""
from evomarks import code_editor

OPERATOR_WEIGHTS = {'tweak_constant': 1.0, 'add_block': 1.0, 'remove_block': 0.5, 'resize_model': 0.7}
EDITS_PER_CHILD = 1
PARENT_SELECTION = 'tournament'
TOURNAMENT_K = 3
STEP_SCALE = 0.3


def choose_parent(archive, rng):
    """Pick the accepted version to mutate next. `archive` is a list of record dicts."""
    pool = [r for r in archive if r['accepted']]
    if PARENT_SELECTION == 'best':
        return max(pool, key=lambda r: r['dev_acc'])
    if PARENT_SELECTION == 'recent':
        return pool[-1]
    entrants = rng.sample(pool, min(TOURNAMENT_K, len(pool)))
    return max(entrants, key=lambda r: r['dev_acc'])


def propose_child(parent_source, rng):
    """Apply EDITS_PER_CHILD random edit operators; returns (new_source, description)."""
    names = list(OPERATOR_WEIGHTS)
    weights = [OPERATOR_WEIGHTS[n] for n in names]
    source, notes = parent_source, []
    for _ in range(EDITS_PER_CHILD):
        op = rng.choices(names, weights=weights)[0]
        source, note = code_editor.apply_operator(op, source, rng, STEP_SCALE)
        notes.append(note)
    return source, '; '.join(notes)


def propose_self_edit(own_source, rng):
    """Rewrite this file's own settings. Returns (new_source, description)."""
    return code_editor.edit_improver(own_source, rng)
