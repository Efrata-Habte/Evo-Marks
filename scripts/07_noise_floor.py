"""Step 7: noise floor of the Level 1 gate. Re-evaluates v000 and the best evolved version under several
training seeds with the exact evolution budget, so we can say whether `min_delta` (and the observed
v000 -> best gain) is larger than seed-to-seed noise. Writes results/noise_floor.json and .md."""
import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import ROOT, get_config
from evomarks import evaluator
from evomarks.archive import Archive
from evomarks.utils import save_json

OUT = ROOT / "results" / "noise_floor"


def main() -> None:
    """Evaluate v000 and best under seeds 0..n-1 (paired by seed) and compare to the gate margin."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = get_config(True if args.quick else None)
    best_id = Archive(ROOT / "archive").best()["id"]
    folders = {}
    for vid in ("v000", best_id):
        folders[vid] = OUT / vid
        folders[vid].mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "archive" / vid / "pipeline.py", folders[vid] / "pipeline.py")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {(vid, s): pool.submit(evaluator.run, folders[vid], cfg, s, "evo", "sum", f"seed{s}")
                for vid in folders for s in range(args.seeds)}
        res = {k: j.result() for k, j in jobs.items()}
    bad = {k: v for k, v in res.items() if "error" in v}
    assert not bad, f"evaluations failed: {bad}"
    stats = {}
    for vid in folders:
        for key in ("dev_acc", "meta_acc"):
            xs = np.array([res[(vid, s)][key] for s in range(args.seeds)])
            stats[f"{vid}.{key}"] = {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)), "values": xs.tolist()}
    gain = {}
    for key in ("dev_acc", "meta_acc"):
        d = np.array(stats[f"{best_id}.{key}"]["values"]) - np.array(stats[f"v000.{key}"]["values"])
        gain[key] = {"mean": float(d.mean()), "std": float(d.std(ddof=1)), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                     "wins": int((d > 0).sum()), "n": len(d)}
    save_json({"best_id": best_id, "seeds": args.seeds, "min_delta": cfg.min_delta, "stats": stats, "paired_gain": gain},
              ROOT / "results" / "noise_floor.json")
    lines = [f"Noise floor of the Level 1 gate ({args.seeds} training seeds, evolution budget, dev n={cfg.n_dev} sheets).", "",
             "| version | dev_acc mean +- std | meta_acc mean +- std |", "|---|---|---|"]
    for vid in folders:
        d, m = stats[f"{vid}.dev_acc"], stats[f"{vid}.meta_acc"]
        lines.append(f"| {vid} | {d['mean']:.4f} +- {d['std']:.4f} | {m['mean']:.4f} +- {m['std']:.4f} |")
    lines += ["", f"Gate margin min_delta = {cfg.min_delta}. Seed std of v000 dev_acc = {stats['v000.dev_acc']['std']:.4f} "
              f"({stats['v000.dev_acc']['std'] / cfg.min_delta:.1f}x the margin).", "",
              f"Paired gain {best_id} - v000 (same seed): "]
    for key, g in gain.items():
        lines.append(f"- {key}: {g['mean']:+.4f} +- {g['se']:.4f} (SE), better in {g['wins']}/{g['n']} seeds")
    (ROOT / "results" / "noise_floor.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
