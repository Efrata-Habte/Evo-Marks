"""Code editor operators, safety checks, tamper detection, archive round trip."""
import random
from pathlib import Path

import pytest

from config import ROOT
from evomarks import code_editor as ce
from evomarks import safety
from evomarks.archive import Archive

PIPE = (ROOT / "evolvable" / "pipeline.py").read_text()
IMPROVER = (ROOT / "evolvable" / "improver.py").read_text()


@pytest.mark.parametrize("op", list(ce.OPERATORS))
def test_operator_returns_parseable_different_safe_source(op):
    rng = random.Random(3)
    # add_block first so remove_block has something to remove
    src = ce.add_block(PIPE, rng, 0.3)[0]
    new, desc = ce.OPERATORS[op](src, rng, 0.3)
    assert new != src and desc
    compile(new, "x", "exec")
    assert safety.check(new).ok


def test_improver_self_edits_are_safe_and_different():
    rng = random.Random(0)
    for _ in range(5):
        new, desc = ce.edit_improver(IMPROVER, rng)
        assert new != IMPROVER and safety.check(new, "improver").ok


def test_seed_files_pass_safety():
    assert safety.check(PIPE).ok and safety.check(IMPROVER, "improver").ok


@pytest.mark.parametrize("bad", ["import os\n", "x = open('f')\n", "exec('1')\n", "y = verifier\n", "TEST = 1\n",
                                 "from evomarks import verifier\n", "import subprocess\n"])
def test_safety_rejects_forbidden(bad):
    assert not safety.check(PIPE + bad).ok


def test_tamper_detection_fires(tmp_path, monkeypatch):
    f = ROOT / "tests" / "_tamper_probe.py"
    before = safety.snapshot()
    f.write_text("x = 1\n")
    try:
        with pytest.raises(safety.TamperError):
            safety.assert_unchanged(before)
    finally:
        f.unlink()
    safety.assert_unchanged(before)       # restored -> passes again


def test_archive_round_trip_keeps_parent_links(tmp_path):
    a = Archive(tmp_path)
    v0, _ = a.stage("pipeline", PIPE)
    a.commit(v0, "pipeline", None, "seed", {"dev_acc": 0.9, "meta_acc": 0.9}, True, "root", 0)
    v1, _ = a.stage("pipeline", PIPE + "# x\n")
    a.commit(v1, "pipeline", v0, "edit", {"dev_acc": 0.8}, False, "no_improvement", 1)
    b = Archive(tmp_path)                 # reload from disk
    assert b.get(v1)["parent_id"] == v0 and not b.get(v1)["accepted"]
    assert b.lineage(v1) == [v0, v1] and b.source(v0) == PIPE
