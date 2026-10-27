"""The archive: every version on disk (source, metrics, parent, model), accepted or rejected.

Darwin-Goedel-Machine style: an open-ended record of all candidates, not just the winner.
Layout: <root>/index.json, <root>/v000/pipeline.py (+metrics.json, model.pt if accepted),
<root>/improver_v000/improver.py. Rejected versions are kept and flagged.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

from evomarks.models import Reader
from evomarks.utils import load_json, load_module, now_iso, save_json

FILES = {"pipeline": "pipeline.py", "improver": "improver.py"}
PREFIX = {"pipeline": "v", "improver": "improver_v"}
SUMMARY_KEYS = ("dev_acc", "meta_acc", "sheet_read_rate", "n_params", "train_time_s", "utility", "utility_old")


class Archive:
    """Index + one folder per version."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.items: list[dict] = load_json(self.index_path) if self.index_path.exists() else []

    def stage(self, kind: str, source: str) -> tuple[str, Path]:
        """Reserve the next id and write the source file; the record is added by commit()."""
        n = len(list(self.root.glob(f"{PREFIX[kind]}[0-9]*")))   # folders, so ids never repeat
        vid = f"{PREFIX[kind]}{n:03d}"
        folder = self.root / vid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / FILES[kind]).write_text(source)
        return vid, folder

    def commit(self, vid: str, kind: str, parent_id: str | None, edit: str, metrics: dict | None,
               accepted: bool, reason: str, gen: int) -> dict:
        """Record the decision; drop model.pt of rejected pipelines to save disk."""
        metrics = metrics or {}
        rec = {"id": vid, "parent_id": parent_id, "kind": kind, "edit": edit, "accepted": accepted,
               "reason": reason, "gen": gen, "timestamp": now_iso(),
               **{k: metrics[k] for k in SUMMARY_KEYS if k in metrics}}
        if not accepted and (self.root / vid / "model.pt").exists():
            (self.root / vid / "model.pt").unlink()
        self.items.append(rec)
        save_json(self.items, self.index_path)
        return rec

    def get(self, vid: str) -> dict:
        """Record by id."""
        return next(r for r in self.items if r["id"] == vid)

    def records(self, kind: str | None = None) -> list[dict]:
        """All records (optionally of one kind), oldest first."""
        return [r for r in self.items if kind is None or r["kind"] == kind]

    def source(self, vid: str) -> str:
        """Source text of a version."""
        return (self.root / vid / FILES[self.get(vid)["kind"]]).read_text()

    def metrics(self, vid: str, name: str = "metrics.json") -> dict:
        """Full metrics of a version (confusion, per-digit accuracy, round logs...)."""
        return load_json(self.root / vid / name)

    def lineage(self, vid: str) -> list[str]:
        """Ids from the root to `vid`."""
        chain = [vid]
        while self.get(chain[-1])["parent_id"]:
            chain.append(self.get(chain[-1])["parent_id"])
        return chain[::-1]

    def best(self) -> dict:
        """Accepted pipeline with the highest dev accuracy (earliest wins ties)."""
        return max(self.records("pipeline"), key=lambda r: (r["accepted"], r.get("dev_acc", 0), -r["gen"]))

    def copy_to(self, vid: str, dest: str) -> None:
        """Copy a version folder (used for archive/best)."""
        target = self.root / dest
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(self.root / vid, target)
        (target / "best_id.txt").write_text(vid)


def load_reader(folder: Path, name: str, weights: str = "model.pt") -> Reader:
    """Rebuild a trained teacher from a version folder: its pipeline.py builds the model."""
    pipe = load_module(Path(folder) / "pipeline.py", f"pipe_{abs(hash(str(folder)))}")
    model = pipe.build_teacher()
    model.load_state_dict(torch.load(Path(folder) / weights, map_location="cpu"))
    return Reader(name, model.eval(), pipe.preprocess)


def load_student(path: Path, name: str, hidden: int = 32) -> Reader:
    """Rebuild a distilled student (tiny MLP, normalize-only preprocessing) from a saved state_dict."""
    from evomarks import blocks, models
    model = models.tiny_student(hidden)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return Reader(name, model.eval(), blocks.normalize)


def default_readers(root: Path, results: Path) -> dict[str, Reader]:
    """The three models shown in the demo: v000 (+L0), best evolved teacher, distilled student."""
    out = {}
    if (root / "v000" / "full" / "model.pt").exists():
        out["v000"] = load_reader(root / "v000", "v000", "full/model.pt")
    if (root / "best" / "full" / "model.pt").exists():
        out["best"] = load_reader(root / "best", "best", "full/model.pt")
    stu = results / "students" / "student_distilled_from_best_s0.pt"
    if stu.exists():
        hidden = load_json(results / "distill_results.json").get("student_hidden", 32)
        out["student"] = load_student(stu, "student", hidden)
    return out
