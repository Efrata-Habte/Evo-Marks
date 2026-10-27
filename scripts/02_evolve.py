"""Step 2: Levels 0/1/2. Self-training (L0) inside a self-modifying loop (L1) whose improver
rewrites itself (L2), plus ablations. Only archive/ and results/ are ever written."""
import argparse
import ast
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import ROOT, Config, get_config
from evomarks import evaluator, safety
from evomarks.archive import Archive
from evomarks.utils import load_module, save_json


def start_utility(imp, parent_src: str, work: Path, cfg: Config, seed: int, gen: int, tag: str, pool):
    """Ask improver `imp` for utility_edits children of a FIXED parent and launch their evaluations
    (in parallel: each is an independent sandboxed subprocess). Same rng seeds for old and new
    improver, so the comparison is paired."""
    jobs = []
    for b in range(cfg.utility_edits):
        src, _ = imp.propose_child(parent_src, random.Random(seed * 7919 + gen * 31 + b))
        if src == parent_src or not safety.check(src).ok:
            jobs.append(None)                                  # wasted draw: counts as zero gain
            continue
        folder = work / "_utility" / f"g{gen:02d}_{tag}_{b}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "pipeline.py").write_text(src)
        jobs.append(pool.submit(evaluator.run, folder, cfg, seed))
    return jobs


def collect_utility(jobs: list, parent: dict, cfg: Config):
    """STOP-style utility: mean meta_dev gain over the proposed edits. An edit only counts if the dev
    gate would accept it (regressions are rolled back and count 0); the gain is measured on meta_dev,
    which the gate never saw."""
    gains = []
    for job in jobs:
        m = job.result() if job else {"error": "none"}
        ok = "error" not in m and m["dev_acc"] >= parent["dev_acc"] + cfg.min_delta
        gains.append(m["meta_acc"] - parent["meta_acc"] if ok else 0.0)
    return float(np.mean(gains)), gains


def level1_step(arch: Archive, imp, cfg: Config, seed: int, gen: int) -> None:
    """One generation: propose a child pipeline, safety-check, evaluate, accept or reject (kept either way)."""
    rng = random.Random(seed * 1000 + gen)
    parent = imp.choose_parent(arch.records("pipeline"), rng)
    parent_src = arch.source(parent["id"])
    src, edit = imp.propose_child(parent_src, rng)
    vid, folder = arch.stage("pipeline", src)
    verdict, m, accepted = safety.check(src, "pipeline"), None, False
    if src == parent_src:
        reason = "no_change"
    elif not verdict.ok:
        reason = "unsafe:" + verdict.reason
    else:
        m = evaluator.run(folder, cfg, seed)
        if "error" in m:
            reason, m = m["error"], None
        elif m["dev_acc"] >= parent["dev_acc"] + cfg.min_delta:    # Gate on dev; the improver never sees test
            accepted, reason = True, "ok"
        else:
            reason = "no_improvement"
    arch.commit(vid, "pipeline", parent["id"], edit, m, accepted, reason, gen)
    dev = f"{m['dev_acc']:.4f}" if m else "  n/a "
    print(f"  [{arch.root.name}] gen {gen:02d} {parent['id']}->{vid} dev {dev} (parent {parent['dev_acc']:.4f}) "
          f"{'ACCEPT' if accepted else 'reject:' + reason:22s} {edit}")


def level2_step(arch: Archive, imp_id: str, imp, cfg: Config, seed: int, gen: int):
    """The recursive step: the improver rewrites ITS OWN source; keep it only if it is a better improver."""
    own = arch.source(imp_id)
    new_src, edit = imp.propose_self_edit(own, random.Random(seed * 5000 + gen))
    nid, nfolder = arch.stage("improver", new_src)
    verdict, new_imp, metrics, accepted = safety.check(new_src, "improver"), None, {}, False
    if new_src == own:
        reason = "no_change"
    elif not verdict.ok:
        reason = "unsafe:" + verdict.reason
    else:
        try:
            new_imp = load_module(nfolder / "improver.py", f"imp_{nid}")
            best = arch.best()
            with ThreadPoolExecutor(max_workers=4) as pool:
                psrc = arch.source(best["id"])
                old_jobs = start_utility(imp, psrc, arch.root, cfg, seed, gen, "old", pool)
                new_jobs = start_utility(new_imp, psrc, arch.root, cfg, seed, gen, "new", pool)
                (u_old, g_old), (u_new, g_new) = collect_utility(old_jobs, best, cfg), collect_utility(new_jobs, best, cfg)
            metrics = {"utility": u_new, "utility_old": u_old, "gains_new": g_new, "gains_old": g_old}
            # Gate: the new improver must beat the old one by a margin on meta_dev-measured utility
            accepted = u_new > u_old + cfg.min_delta_l2
            reason = "ok" if accepted else "utility_not_better"
        except Exception as e:                                    # a broken improver is simply rejected
            reason = f"crash:{type(e).__name__}"
    save_json(metrics, nfolder / "metrics.json")
    arch.commit(nid, "improver", imp_id, edit, metrics, accepted, reason, gen)
    print(f"  [{arch.root.name}] L2 gen {gen:02d} {imp_id}->{nid} utility {metrics.get('utility_old', 0):+.4f} -> "
          f"{metrics.get('utility', 0):+.4f} {'ACCEPT' if accepted else 'reject:' + reason} | {edit}")
    return (nid, new_imp) if accepted else (imp_id, imp)


def normalized(name: str) -> str:
    """Seed file as ast.unparse prints it, so every later diff shows only real edits (no whitespace noise)."""
    return ast.unparse(ast.parse((ROOT / "evolvable" / name).read_text())) + "\n"


def run_evolution(root: Path, cfg: Config, seed: int, level2: bool) -> Archive:
    """Full evolution run from the seed files in evolvable/. Wipes and rebuilds `root`."""
    assert any(root.is_relative_to(r) for r in safety.EDITABLE_ROOTS), "may only write under archive/ or results/"
    shutil.rmtree(root, ignore_errors=True)
    arch = Archive(root)
    vid, folder = arch.stage("pipeline", normalized("pipeline.py"))
    m = evaluator.run(folder, cfg, seed)
    assert "error" not in m, f"seed pipeline failed: {m}"
    arch.commit(vid, "pipeline", None, "seed version (evolvable/pipeline.py)", m, True, "root", 0)
    imp_id, ifolder = arch.stage("improver", normalized("improver.py"))
    arch.commit(imp_id, "improver", None, "seed improver (evolvable/improver.py)", {}, True, "root", 0)
    imp = load_module(ifolder / "improver.py", "imp_seed")
    print(f"{root.name}: v000 dev={m['dev_acc']:.4f} meta={m['meta_acc']:.4f} level2={level2}")
    for gen in range(1, cfg.generations + 1):
        level1_step(arch, imp, cfg, seed, gen)
        if level2 and gen % cfg.improver_every == 0 and gen < cfg.generations:
            imp_id, imp = level2_step(arch, imp_id, imp, cfg, seed, gen)
    shutil.rmtree(root / "_utility", ignore_errors=True)
    return arch


def history(arch: Archive) -> list[dict]:
    """Best-so-far dev (and its meta_dev) after each generation, for the ablation chart."""
    best, out = None, []
    for r in arch.records("pipeline"):
        if r["accepted"] and (best is None or r["dev_acc"] > best["dev_acc"]):
            best = r
        out.append({"gen": r["gen"], "best_dev": best["dev_acc"], "best_meta": best["meta_acc"], "id": r["id"]})
    return out


def finalize(arch: Archive, cfg: Config, seed: int) -> None:
    """Retrain v000 and the best version at FULL budget (sum-check, and the confidence ablation).
    The four runs are independent sandboxed subprocesses, so they run side by side."""
    best = arch.best()
    jobs = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for vid in dict.fromkeys(["v000", best["id"]]):
            for mode, out in (("sum", "full"), ("confidence", "full_conf")):
                jobs[(vid, mode)] = pool.submit(evaluator.run, arch.root / vid, cfg, seed, "full", mode, out)
    for (vid, mode), job in jobs.items():
        m = job.result()
        print(f"  full[{mode:10s}] {vid}: dev={m.get('dev_acc', float('nan')):.4f} meta={m.get('meta_acc', float('nan')):.4f}")
    arch.copy_to(best["id"], "best")


def main() -> None:
    """Main run (Level 1 + Level 2), then the Level-1-only ablation and optional extra seeds."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    cfg = get_config(True if args.quick else None)
    t0 = time.time()
    abl = ROOT / "results" / "ablation"
    with ThreadPoolExecutor(max_workers=6) as pool:        # independent runs share nothing but read-only data
        main_job = pool.submit(run_evolution, ROOT / "archive", cfg, cfg.seed, True)
        jobs = {"l1_only": [], "l1_l2": []}
        if not args.no_ablation:
            for k in range(cfg.evo_seeds):
                if k:
                    jobs["l1_l2"].append(pool.submit(run_evolution, abl / f"l1_l2_s{k}", cfg, cfg.seed + k, True))
                jobs["l1_only"].append(pool.submit(run_evolution, abl / f"l1_only_s{k}", cfg, cfg.seed + k, False))
        main_arch = main_job.result()
        finalize(main_arch, cfg, cfg.seed)
    runs = {"l1_l2": [history(main_arch)] + [history(j.result()) for j in jobs["l1_l2"]],
            "l1_only": [history(j.result()) for j in jobs["l1_only"]]}
    best = main_arch.best()
    save_json({"quick": cfg.quick, "best_id": best["id"], "generations": cfg.generations, "runs": runs,
               "tamper": evaluator.TAMPER, "runtime_s": time.time() - t0}, ROOT / "results" / "evolution_summary.json")
    print(f"best={best['id']} dev={best['dev_acc']:.4f}  tamper checks={evaluator.TAMPER}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
