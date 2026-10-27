"""Step 6: results table (csv + md) and results/analysis_report.md.
Every sentence is computed from the saved numbers; verdict words come from a simple 2-standard-error
rule, so "no clear effect" is reported whenever the evidence is not there."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import ROOT
from evomarks.utils import load_json

ARCH, OUT = ROOT / "archive", ROOT / "results"


def se(p: float, n: int) -> float:
    """Binomial standard error of an accuracy p measured on n digits."""
    return float(np.sqrt(max(p * (1 - p), 1e-9) / n))


def verdict(delta: float, err: float) -> str:
    """helped / no clear effect / hurt, using a 2-standard-error rule."""
    return "helped" if delta > 2 * err else "hurt" if delta < -2 * err else "no clear effect (within 2 standard errors)"


def results_table(kd: dict) -> pd.DataFrame:
    """One row per model; students are mean +- std over seeds."""
    by = defaultdict(list)
    for r in kd["rows"]:
        by[r["name"]].append(r)
    rows = []
    for name, rs in by.items():
        f = lambda k, d=4: (f"{np.mean([r[k] for r in rs]):.{d}f}" + (f" +- {np.std([r[k] for r in rs]):.{d}f}" if len(rs) > 1 else ""))
        rows.append({"model": name, "digit_acc (test)": f("digit_acc"), "sheet_exact_read": f("exact_read_rate"),
                     "flag_precision": f("flag_precision"), "flag_recall": f("flag_recall"), "params": int(rs[0]["params"]),
                     "size_KB": f"{rs[0]['size_kb']:.1f}", "latency_ms/sheet": f("latency_ms", 2), "seeds": len(rs)})
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    return "\n".join([head] + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values])


def level0_section(v0: dict, best: dict, v0c: dict, bc: dict, n_dev: int) -> str:
    """Observations about verified self-training and the verifier-vs-confidence ablation."""
    out = ["## Level 0: verified self-training", ""]
    for name, m, c in (("v000", v0, v0c), ("best", best, bc)):
        rs = m["rounds"]
        gain = m["dev_acc"] - m["baseline_dev_acc"]
        v = m["verifier_stats"]
        prec = sum((r["verifier_precision"] or 0) * r["accepted_digits"] for r in rs) / max(sum(r["accepted_digits"] for r in rs), 1)
        cp = c["verifier_stats"]
        cd = sum(r["accepted_digits"] for r in c["rounds"])
        cprec = f"{sum((r['verifier_precision'] or 0) * r['accepted_digits'] for r in c['rounds']) / cd:.4f}" if cd else "n/a"
        out += [f"- **{name}**: dev accuracy {m['baseline_dev_acc']:.4f} -> {m['dev_acc']:.4f} ({gain:+.4f}; "
                f"{verdict(gain, se(m['dev_acc'], n_dev))}). {sum(r['decision'] == 'accept' for r in rs)} of {len(rs)} rounds accepted, "
                f"{sum(r['decision'] == 'reject' for r in rs)} rolled back.",
                f"  - sum check accepted {v['sheets_accepted']}/{v['sheets_read']} sheets; pseudo-label precision {prec:.4f}; "
                f"{v['compensating_errors']} accepted sheets contained compensating errors; "
                f"{v['low_conf_accepted']} accepted digits were low-confidence (<0.6), of which {v['low_conf_correct']} were correct.",
                f"  - confidence-threshold ablation (each digit >= 0.9): kept {cd} digits (sum check kept {v['digits_accepted']}), "
                f"pseudo-label precision {cprec} (sum check {prec:.4f}), "
                f"final dev accuracy {c['dev_acc']:.4f} vs {m['dev_acc']:.4f} for the sum check "
                f"({'sum check better' if m['dev_acc'] > c['dev_acc'] else 'confidence threshold at least as good'})."]
    return "\n".join(out)


def level1_section(index: list, summary: dict, n_dev: int) -> str:
    """Observations about code evolution."""
    pipes = {r["id"]: r for r in index if r["kind"] == "pipeline"}
    best = pipes[summary["best_id"]]
    v0 = pipes["v000"]
    kids = [r for r in pipes.values() if r["parent_id"]]
    reasons = defaultdict(int)
    for r in kids:
        reasons[r["reason"].split(":")[0]] += not r["accepted"]
    ops = defaultdict(list)
    for r in kids:
        if r.get("dev_acc") is not None:
            ops[r["edit"].split(":")[0]].append((r["dev_acc"] - pipes[r["parent_id"]]["dev_acc"], r["accepted"]))
    d, dm = best["dev_acc"] - v0["dev_acc"], best["meta_acc"] - v0["meta_acc"]
    out = ["## Level 1: self-modifying code", "",
           f"- {len(kids)} candidate versions in {summary['generations']} generations; {sum(r['accepted'] for r in kids)} accepted, "
           f"{sum(not r['accepted'] for r in kids)} rejected ({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))}).",
           f"- Best version {best['id']}: dev {v0['dev_acc']:.4f} -> {best['dev_acc']:.4f} ({d:+.4f}); the dev gain is optimistic because "
           f"{best['id']} was *selected* on dev. On meta_dev (not used for this selection) {v0['meta_acc']:.4f} -> {best['meta_acc']:.4f} "
           f"({dm:+.4f}, {verdict(dm, se(best['meta_acc'], n_dev))}).",
           "- Edit effectiveness (mean change in dev accuracy vs parent):"]
    out += [f"  - {k}: n={len(v)}, accepted {sum(a for _, a in v)}, mean delta {np.mean([x for x, _ in v]):+.4f}" for k, v in sorted(ops.items())]
    return "\n".join(out)


def level2_section(index: list, summary: dict) -> str:
    """Observations about the recursive step and the Level 1 vs Level 1+2 ablation."""
    imps = [r for r in index if r["kind"] == "improver" and r["parent_id"]]
    out = ["## Level 2: the improver improving itself", "",
           f"- {len(imps)} self-edit proposals, {sum(r['accepted'] for r in imps)} accepted."]
    out += [f"  - gen {r['gen']}: {r['edit']} -> utility {r.get('utility_old', 0):+.4f} -> {r.get('utility', 0):+.4f} "
            f"({'kept' if r['accepted'] else 'rolled back: ' + r['reason']})" for r in imps]
    final = {k: [h[-1]["best_dev"] for h in v] for k, v in summary["runs"].items() if v}
    if len(final) == 2:
        a, b = np.mean(final["l1_l2"]), np.mean(final["l1_only"])
        out.append(f"- Ablation (best dev accuracy after the last generation, mean over {len(final['l1_l2'])} seed(s)): "
                   f"Level 1 + 2 = {a:.4f}, Level 1 only = {b:.4f} (difference {a - b:+.4f}). With so few runs, "
                   "a difference this small cannot be separated from evaluation noise unless it exceeds ~2 standard errors.")
    return "\n".join(out)


def distill_section(kd: dict, n_test: int) -> str:
    """Observations about distillation."""
    by = defaultdict(list)
    for r in kd["rows"]:
        by[r["name"]].append(r)
    acc = lambda n: np.array([r["digit_acc"] for r in by[n]])
    t, h = by["teacher best (evolved)"][0], acc("student hard-label")
    dv, db = acc("student distilled <- v000"), acc("student distilled <- best")
    s0 = by["student hard-label"][0]
    out = ["## Distillation", "",
           f"- Chosen on dev: {', '.join(f'{k}: T={v['T']:g}, alpha={v['alpha']:g}' for k, v in kd['chosen'].items())}.",
           f"- Test digit accuracy (mean +- std over 3 seeds): hard-label student {h.mean():.4f} +- {h.std():.4f}; "
           f"distilled from v000 {dv.mean():.4f} +- {dv.std():.4f}; distilled from best {db.mean():.4f} +- {db.std():.4f}; "
           f"teacher best {t['digit_acc']:.4f}.",
           f"- Distillation vs hard-label (best teacher): {db.mean() - h.mean():+.4f} -> "
           f"{verdict(db.mean() - h.mean(), max(h.std(), db.std(), se(h.mean(), n_test)))}.",
           f"- Size/speed: student {s0['params']} params / {s0['size_kb']:.0f} KB / {s0['latency_ms']:.2f} ms per sheet vs teacher "
           f"{t['params']} params / {t['size_kb']:.0f} KB / {t['latency_ms']:.2f} ms ({t['params'] / s0['params']:.1f}x fewer params, "
           f"{t['latency_ms'] / s0['latency_ms']:.1f}x faster).",
           f"- Teacher minus distilled student, test digit accuracy: {t['digit_acc'] - db.mean():+.4f}."]
    gap = np.mean([r["per_digit_acc"] for r in by["teacher best (evolved)"]], 0) - np.mean([r["per_digit_acc"] for r in by["student distilled <- best"]], 0)
    out.append(f"- Largest per-digit gap (teacher - distilled student): digit {int(gap.argmax())} ({gap.max():+.3f}); the average hides this tail.")
    return "\n".join(out)


def main() -> None:
    """Write results_table.csv/.md and analysis_report.md."""
    kd = load_json(OUT / "distill_results.json")
    summary, index = load_json(OUT / "evolution_summary.json"), load_json(ARCH / "index.json")
    v0, best = load_json(ARCH / "v000/full/metrics.json"), load_json(ARCH / "best/full/metrics.json")
    v0c, bc = load_json(ARCH / "v000/full_conf/metrics.json"), load_json(ARCH / "best/full_conf/metrics.json")
    sizes = load_json(ROOT / "data" / "generated" / "meta.json")["sizes"]
    n_dev, n_test = sizes["dev"] * 7, sizes["test"] * 7
    df = results_table(kd)
    df.to_csv(OUT / "results_table.csv", index=False)
    table_md = to_md(df)
    (OUT / "results_table.md").write_text(table_md + "\n")
    figs = lambda *names: "\n".join(f"![{n}](figures/{n}.png)" for n in names if (OUT / "figures" / f"{n}.png").exists())
    body = ["# EvoMarks analysis report", "",
            f"Mode: {'quick' if kd['quick'] else 'full'}. All numbers below are read from `results/` and `archive/`; "
            "dev/meta_dev numbers were used for selection, test numbers are reporting only.", "", "## Results table (test split)", "", table_md, "",
            level0_section(v0, best, v0c, bc, n_dev), "", figs("l0_accuracy_by_round", "l0_accepted_rejected", "l0_verifier_precision",
                                                              "l0_hard_examples", "l0_verifier_vs_confidence", "l0_confusion_delta"), "",
            level1_section(index, summary, n_dev), "", figs("l1_evolution_timeline", "l1_lineage_tree", "l1_edit_effectiveness",
                                                          "l1_rejection_reasons", "l1_best_vs_parent"), "",
            level2_section(index, summary), "", figs("l2_improver_utility", "l2_improver_params", "l2_ablation"), "",
            distill_section(kd, n_test), "", figs("kd_accuracy_vs_latency", "kd_soft_targets", "kd_tsweep_heatmap", "kd_per_digit_gap"), "",
            "## Whole system", "", figs("system_sheet_metrics", "final_real_sheets"), ""]
    (OUT / "analysis_report.md").write_text("\n".join(body))
    print("wrote results_table.csv/.md and analysis_report.md")


if __name__ == "__main__":
    main()
