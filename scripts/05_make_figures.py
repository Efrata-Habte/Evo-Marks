"""Step 5: static PNG figures for the slides. Each title is the question the figure answers.
Reads only archive/ and results/ (never trains, never selects anything)."""
import ast
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import ROOT
from evomarks.utils import load_json

ARCH, OUT = ROOT / "archive", ROOT / "results"
FIG = OUT / "figures"
C = {"acc": "#2a9d5c", "rej": "#9aa0a6", "bad": "#d1495b", "teacher": "#3a6ea5", "student": "#e08a2c",
     "v000": "#7a8896", "best": "#3a6ea5", "ink": "#222222"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def save(fig, name: str) -> None:
    """Write results/figures/<name>.png."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def runs(ctx):
    """The two full-budget Level 0 runs: v000 (before evolution) and the best evolved version."""
    return [("v000", ctx["v0"]), ("best", ctx["best"])]


# ---------------- Level 0 ----------------
def l0_accuracy_by_round(ctx):
    """Dev accuracy per round, accepted vs rejected candidates marked."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, m in runs(ctx):
        xs = [0] + [r["round"] for r in m["rounds"]]
        ax.plot(xs, [m["baseline_dev_acc"]] + [r["dev_acc"] for r in m["rounds"]], "-", color=C[name], label=f"teacher ({name})")
        for r in m["rounds"]:
            ok = r["decision"] == "accept"
            ax.scatter(r["round"], r["candidate_dev_acc"], s=60, marker="o" if ok else "X",
                       facecolors=C["acc"] if ok else "none", edgecolors=C["acc"] if ok else C["bad"], zorder=3)
    ax.set(xlabel="self-training round (0 = seed-only teacher)", ylabel="dev digit accuracy",
           title="Did learning from verified sheets help?")
    ax.text(0.02, 0.04, "green dot = candidate accepted, red X = rejected (rolled back)\n1 seed per curve", transform=ax.transAxes, fontsize=8)
    ax.set_xticks(range(max(len(m["rounds"]) for _, m in runs(ctx)) + 1))
    ax.legend(loc="lower right")
    save(fig, "l0_accuracy_by_round")


def l0_accepted_rejected(ctx):
    """Accepted / rejected sheets per round, by rejection reason."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (name, m) in zip(axes, runs(ctx)):
        rs = m["rounds"]
        a = np.array([r["accepted"] for r in rs])
        mx = np.array([r["rejected"].get("mark_exceeds_max", 0) for r in rs])
        sm = np.array([r["rejected"].get("sum_mismatch", 0) for r in rs])
        x = np.arange(1, len(rs) + 1)
        ax.bar(x, a, color=C["acc"], label="accepted")
        ax.bar(x, sm, bottom=a, color=C["rej"], label="rejected: sum_mismatch")
        ax.bar(x, mx, bottom=a + sm, color=C["bad"], label="rejected: mark_exceeds_max")
        ax.set(title=f"{name}", xlabel="round", xticks=x)
    axes[0].set_ylabel("sheets read")
    axes[0].legend(fontsize=8)
    fig.suptitle("How many sheets does the sum check let through?", fontweight="bold")
    save(fig, "l0_accepted_rejected")


def l0_verifier_precision(ctx):
    """Share of accepted digits whose pseudo-label is right, and compensating errors."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax2 = ax.twinx()
    for k, (name, m) in enumerate(runs(ctx)):
        rs = [r for r in m["rounds"] if r["verifier_precision"] is not None]
        x = np.array([r["round"] for r in rs])
        ax.plot(x, [r["verifier_precision"] for r in rs], "o-", color=C[name], label=f"precision ({name})")
        ax2.bar(x + (k - 0.5) * 0.3, [r["compensating_errors"] for r in rs], 0.3, color=C[name], alpha=0.35,
                label=f"compensating errors ({name})")
    ax.set(xlabel="round", ylabel="verifier precision (accepted digits that are correct)", ylim=(0.9, 1.005),
           title="How often does the sum check accept a wrong label?")
    ax2.set_ylabel("compensating-error sheets"), ax2.grid(False)
    ax.legend(loc="lower left", fontsize=8), ax2.legend(loc="lower right", fontsize=8)
    save(fig, "l0_verifier_precision")


def l0_hard_examples(ctx):
    """Low-confidence digits rescued by the sum check + sheets rejected for sum_mismatch.
    Shows the FIRST examples collected (no cherry-picking)."""
    z = np.load(ARCH / "best" / "full" / "l0_examples.npz")
    n, m = min(24, len(z["r_img"])), min(4, len(z["j_img"]))
    fig = plt.figure(figsize=(9, 8))
    top, bottom = fig.add_gridspec(2, 1, height_ratios=[3, 2.2], hspace=0.25)
    g1 = top.subgridspec(3, 8, hspace=0.5, wspace=0.1)
    g2 = bottom.subgridspec(max(m, 1), 7, hspace=0.6, wspace=0.1)
    for i in range(n):
        ax = fig.add_subplot(g1[i // 8, i % 8])
        ax.imshow(z["r_img"][i], cmap="gray"), ax.axis("off")
        good = z["r_pred"][i] == z["r_true"][i]
        ax.set_title(f"{z['r_pred'][i]} ({z['r_conf'][i]:.2f})", fontsize=8, color=C["acc"] if good else C["bad"])
        if i == 3:
            ax.annotate("Which hard digits did the sum check rescue? (teacher guess, confidence < 0.6; green = guess right)",
                        (0, 1.6), xycoords="axes fraction", fontsize=10, fontweight="bold")
    for r in range(m):
        for j in range(7):
            ax = fig.add_subplot(g2[r, j])
            ax.imshow(z["j_img"][r, j], cmap="gray"), ax.set_xticks([]), ax.set_yticks([])
            wrong = z["j_pred"][r, j] != z["j_true"][r, j]
            for sp in ax.spines.values():
                sp.set_edgecolor(C["bad"] if wrong else C["rej"]), sp.set_linewidth(3 if wrong else 0.5)
            ax.set_title(f"read {z['j_pred'][r, j]}", fontsize=8, color=C["bad"] if wrong else C["ink"])
            if r == 0 and j == 0:
                ax.annotate("Sheets rejected for sum_mismatch (red = misread digit; no red = total genuinely wrong)",
                            (0, 1.55), xycoords="axes fraction", fontsize=10, fontweight="bold")
    save(fig, "l0_hard_examples")


def l0_verifier_vs_confidence(ctx):
    """Ablation: sum-check verifier vs confidence-threshold pseudo-labels."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for k, (name, a, b) in enumerate((("v000", ctx["v0"], ctx["v0conf"]), ("best", ctx["best"], ctx["bestconf"]))):
        for j, (lab, m) in enumerate((("sum check", a), ("confidence >= 0.9", b))):
            rs = [r for r in m["rounds"] if r["verifier_precision"] is not None]
            w = sum(r["accepted_digits"] for r in rs) or 1
            prec = sum(r["verifier_precision"] * r["accepted_digits"] for r in rs) / w
            for ax, v in zip(axes, (prec, m["verifier_stats"]["digits_accepted"], m["dev_acc"])):
                ax.bar(k * 3 + j, v, color=C["acc"] if j == 0 else C["rej"], label=lab if k == 0 else None)
    for ax, t in zip(axes, ("pseudo-label precision", "digits kept as pseudo-labels", "final dev accuracy")):
        ax.set(title=t, xticks=[0.5, 3.5], xticklabels=["v000", "best"])
    axes[0].set_ylim(0.9, 1.0), axes[2].set_ylim(0.7, 1.0)
    axes[0].legend(fontsize=8)
    fig.suptitle("Is arithmetic a better judge than the model's own confidence?", fontweight="bold")
    save(fig, "l0_verifier_vs_confidence")


def l0_confusion_delta(ctx):
    """Dev confusion matrix after - before self-training (best version)."""
    m = ctx["best"]
    d = np.array(m["confusion"]) - np.array(m["baseline_confusion"])
    fig, ax = plt.subplots(figsize=(5.8, 5))
    v = max(abs(d).max(), 1)
    im = ax.imshow(d, cmap="RdBu", vmin=-v, vmax=v)
    for i in range(10):
        for j in range(10):
            if d[i, j]:
                ax.text(j, i, d[i, j], ha="center", va="center", fontsize=7)
    ax.set(xlabel="predicted", ylabel="true", xticks=range(10), yticks=range(10), title="Which confusions did the loop fix?")
    ax.grid(False)
    fig.colorbar(im, label="change in count (blue diagonal = more correct, red off-diagonal = more errors)")
    save(fig, "l0_confusion_delta")


# ---------------- Level 1 ----------------
def pipeline_records(ctx):
    """All pipeline records plus helper dict."""
    return [r for r in ctx["index"] if r["kind"] == "pipeline"]


def l1_evolution_timeline(ctx):
    """Dev accuracy of every generated version; accepted filled, rejected hollow."""
    recs, fig = pipeline_records(ctx), plt.figure(figsize=(7.5, 4))
    ax, best, xs, ys = fig.gca(), -1, [], []
    for r in recs:
        if r.get("dev_acc") is None:
            ax.scatter(r["gen"], ax.get_ylim()[0], marker="x", color=C["bad"])
            continue
        ax.scatter(r["gen"], r["dev_acc"], s=55, marker="o", facecolors=C["acc"] if r["accepted"] else "none",
                   edgecolors=C["acc"] if r["accepted"] else C["rej"], zorder=3)
        best = max(best, r["dev_acc"]) if r["accepted"] else best
        xs.append(r["gen"]), ys.append(best)
    ax.step(xs, ys, where="post", color=C["best"], label="best accepted so far")
    ax.set(xlabel="generation", ylabel="dev digit accuracy (evolution budget)", title="Did self-edited code get better over generations?")
    ax.legend()
    save(fig, "l1_evolution_timeline")


def l1_lineage_tree(ctx):
    """Parent -> child tree of all versions."""
    recs = {r["id"]: r for r in pipeline_records(ctx)}
    kids = defaultdict(list)
    for r in recs.values():
        if r["parent_id"]:
            kids[r["parent_id"]].append(r["id"])
    ypos, counter = {}, [0]

    def walk(v):
        """Depth-first: leaves get consecutive rows."""
        if not kids[v]:
            ypos[v] = counter[0]
            counter[0] += 1
        else:
            for c in kids[v]:
                walk(c)
            ypos[v] = np.mean([ypos[c] for c in kids[v]])
    walk("v000")
    acc = [r["dev_acc"] for r in recs.values() if r["accepted"]]
    norm = plt.Normalize(min(acc) - 0.005, max(acc))
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.5 * counter[0]))
    for r in recs.values():
        if r["parent_id"]:
            p = recs[r["parent_id"]]
            ax.plot([p["gen"], r["gen"]], [ypos[p["id"]], ypos[r["id"]]], color="#bbbbbb", zorder=1)
    for r in recs.values():
        col = plt.cm.viridis(norm(r["dev_acc"])) if r["accepted"] else "#dddddd"
        ax.scatter(r["gen"], ypos[r["id"]], s=300, color=col, edgecolors="k" if r["accepted"] else "#999999", zorder=2)
        ax.text(r["gen"], ypos[r["id"]], r["id"][1:], ha="center", va="center", fontsize=7, color="w" if r["accepted"] else "#555")
    ax.set(xlabel="generation", yticks=[], title="Which versions descend from which? (colour = dev accuracy, grey = rejected)")
    ax.grid(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="dev accuracy (accepted)")
    save(fig, "l1_lineage_tree")


def l1_edit_effectiveness(ctx):
    """Mean change in dev accuracy per edit operator and per block."""
    recs = {r["id"]: r for r in pipeline_records(ctx)}
    by_op, by_block = defaultdict(list), defaultdict(list)
    for r in recs.values():
        if r["parent_id"] and r.get("dev_acc") is not None:
            d = r["dev_acc"] - recs[r["parent_id"]]["dev_acc"]
            by_op[r["edit"].split(":")[0]].append(d)
            if r["edit"].startswith(("add_block", "remove_block")):
                by_block[r["edit"].split(": ")[1]].append(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, groups, t in zip(axes, (by_op, by_block), ("edit operator", "block edited")):
        keys = sorted(groups)
        means = [np.mean(groups[k]) for k in keys]
        ax.barh(keys, means, color=[C["acc"] if v > 0 else C["bad"] for v in means])
        for i, k in enumerate(keys):
            ax.text(0, i, f"  n={len(groups[k])}", va="center", fontsize=8)
        ax.set(xlabel="mean change in dev accuracy vs parent", title=t)
    fig.suptitle("Which self-edits actually helped?", fontweight="bold")
    save(fig, "l1_edit_effectiveness")


def l1_rejection_reasons(ctx):
    """Why candidates were rejected."""
    cnt = defaultdict(int)
    for r in pipeline_records(ctx):
        if not r["accepted"]:
            cnt[r["reason"].split(":")[0]] += 1
    fig, ax = plt.subplots(figsize=(6, 3.5))
    keys = sorted(cnt, key=cnt.get)
    ax.barh(keys or ["(none)"], [cnt[k] for k in keys] or [0], color=C["rej"])
    ax.set(xlabel="rejected versions", title="Why were self-edits rejected? (unsafe / crash / timeout / no gain)")
    save(fig, "l1_rejection_reasons")


def l1_best_vs_parent(ctx):
    """Per-digit dev accuracy: best version vs its parent vs v000."""
    recs = {r["id"]: r for r in pipeline_records(ctx)}
    best = ctx["summary"]["best_id"]
    parent = recs[best]["parent_id"] or "v000"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, (vid, col) in enumerate((("v000", C["v000"]), (parent, C["rej"]), (best, C["best"]))):
        pd_acc = load_json(ARCH / vid / "metrics.json")["per_digit_acc"]
        ax.bar(np.arange(10) + (k - 1) * 0.27, pd_acc, 0.27, color=col, label=f"{vid}{' (best)' if vid == best else ''}")
    ax.set(xticks=range(10), xlabel="digit", ylabel="dev accuracy", ylim=(0.6, 1.0),
           title="Where did the best version improve over its parent and v000?")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "l1_best_vs_parent")


# ---------------- Level 2 ----------------
def l2_improver_utility(ctx):
    """Utility of old vs proposed improver at each recursive step."""
    imps = [r for r in ctx["index"] if r["kind"] == "improver" and r["parent_id"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, r in enumerate(imps):
        ax.bar(i - 0.2, r.get("utility_old", 0), 0.4, color=C["rej"], label="current improver" if i == 0 else None)
        ax.bar(i + 0.2, r.get("utility", 0), 0.4, color=C["acc"] if r["accepted"] else C["bad"],
               label="proposed (green = kept, red = rolled back)" if i == 0 else None)
    ax.set(xticks=range(len(imps)), xticklabels=[f"gen {r['gen']}" for r in imps], ylabel="utility (mean meta_dev gain per edit)",
           title="Did the improver's self-edit make it a better improver?")
    if imps:
        ax.legend(fontsize=8)
    save(fig, "l2_improver_utility")


def l2_improver_params(ctx):
    """How the improver's own operator weights changed across kept versions."""
    chain = [r["id"] for r in ctx["index"] if r["kind"] == "improver" and r["accepted"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    series = defaultdict(list)
    for vid in chain:
        for node in ast.parse((ARCH / vid / "improver.py").read_text()).body:
            if isinstance(node, ast.Assign) and node.targets[0].id == "OPERATOR_WEIGHTS":
                for k, v in ast.literal_eval(node.value).items():
                    series[k].append(v)
    for k, v in series.items():
        ax.plot(range(len(v)), v, "o-", label=k)
    ax.set(xticks=range(len(chain)), xticklabels=chain, ylabel="operator weight", title="How did the improver rewrite itself?")
    ax.tick_params(axis="x", labelrotation=30)
    if len(chain) == 1:
        ax.text(0.5, 0.5, "no improver self-edit was accepted", transform=ax.transAxes, ha="center", color=C["bad"])
    ax.legend(fontsize=8)
    save(fig, "l2_improver_params")


def l2_ablation(ctx):
    """Level 1 only vs Level 1 + Level 2: best dev accuracy by generation."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, lab, col in (("l1_l2", "Level 1 + Level 2", C["best"]), ("l1_only", "Level 1 only", C["rej"])):
        hs = ctx["summary"]["runs"][key]
        if not hs:
            continue
        for h in hs:
            ax.plot([g["gen"] for g in h], [g["best_dev"] for g in h], color=col, alpha=0.3 if len(hs) > 1 else 1)
        L = min(len(h) for h in hs)
        ax.plot(range(L), np.mean([[g["best_dev"] for g in h[:L]] for h in hs], axis=0), color=col, lw=2.5,
                label=f"{lab} (n={len(hs)} seed{'s' if len(hs) > 1 else ''})")
    ax.set(xlabel="generation", ylabel="best dev accuracy so far", title="Does letting the improver edit itself help?")
    ax.legend()
    save(fig, "l2_ablation")


# ---------------- Distillation & system ----------------
def agg_rows(ctx):
    """Mean/std over seeds for every named model."""
    by = defaultdict(list)
    for r in ctx["kd"]["rows"]:
        by[r["name"]].append(r)
    return by


def kd_accuracy_vs_latency(ctx):
    """Accuracy vs CPU latency, bubble = model size."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k, (name, rs) in enumerate(agg_rows(ctx).items()):
        kind = rs[0]["kind"]
        acc = [r["digit_acc"] for r in rs]
        ax.errorbar(np.mean([r["latency_ms"] for r in rs]), np.mean(acc), yerr=np.std(acc), fmt="none", ecolor="k", capsize=3)
        ax.scatter(np.mean([r["latency_ms"] for r in rs]), np.mean(acc), s=rs[0]["size_kb"] * 4, alpha=0.7,
                   color=C[kind], edgecolors="k")
        ax.annotate(name, (np.mean([r["latency_ms"] for r in rs]), np.mean(acc)), fontsize=7, xytext=(8, -4 - 9 * (k % 4)), textcoords="offset points")
    ax.set(xlabel="CPU latency per sheet (ms, 1 thread)", ylabel="test digit accuracy",
           title="What do we lose by shrinking the model?  (bubble = size in KB, bars = std over 3 seeds)")
    save(fig, "kd_accuracy_vs_latency")


def kd_soft_targets(ctx):
    """Teacher probabilities at T=1 vs T=4 for a 4-looks-like-9 and a 3-looks-like-8."""
    z = np.load(OUT / "kd_softtargets.npz")
    lg, lab = z["logits"], z["labels"]
    sm = lambda t: np.exp(lg / t) / np.exp(lg / t).sum(1, keepdims=True)
    p1 = sm(1.0)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for row, (d, other) in enumerate(((4, 9), (3, 8))):
        cand = np.where((lab == d) & (lg.argmax(1) == d))[0]
        i = cand[np.argmax(p1[cand, other])]
        axes[row, 0].imshow(z["images"][i], cmap="gray"), axes[row, 0].axis("off")
        axes[row, 0].set_title(f"a '{d}' that looks a bit like '{other}'")
        for col, T in ((1, 1.0), (2, 4.0)):
            p = sm(T)[i]
            axes[row, col].bar(range(10), p, color=[C["student"] if k == other else C["teacher"] for k in range(10)])
            axes[row, col].set(title=f"T = {T:g}", xticks=range(10), ylim=(0, 1))
    fig.suptitle("What do soft targets reveal that hard labels hide? (orange = the look-alike class)", fontweight="bold")
    save(fig, "kd_soft_targets")


def kd_tsweep_heatmap(ctx):
    """Dev accuracy over the T x alpha grid for both teachers."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (tk, rows) in zip(axes, ctx["kd"]["sweep"].items()):
        Ts, As = sorted({r["T"] for r in rows}), sorted({r["alpha"] for r in rows})
        grid = np.array([[next(r["dev_acc"] for r in rows if r["T"] == T and r["alpha"] == a) for a in As] for T in Ts])
        im = ax.imshow(grid, cmap="viridis")
        for i in range(len(Ts)):
            for j in range(len(As)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
        ax.set(xticks=range(len(As)), xticklabels=As, yticks=range(len(Ts)), yticklabels=Ts, xlabel="alpha", ylabel="T",
               title=f"teacher {tk}")
        ax.grid(False)
    fig.suptitle("Which temperature / alpha works best? (student dev accuracy)", fontweight="bold")
    save(fig, "kd_tsweep_heatmap")


def kd_per_digit_gap(ctx):
    """Per-digit test accuracy: teacher vs distilled vs hard-label student."""
    by = agg_rows(ctx)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, (name, col) in enumerate((("teacher best (evolved)", C["teacher"]), ("student distilled <- best", C["student"]),
                                     ("student hard-label", C["rej"]))):
        ax.bar(np.arange(10) + (k - 1) * 0.27, np.mean([r["per_digit_acc"] for r in by[name]], axis=0), 0.27, color=col, label=name)
    ax.set(xticks=range(10), xlabel="digit", ylabel="test accuracy", ylim=(0.6, 1.0),
           title="Where do students lag the teacher? (averages hide the tail)")
    ax.legend(fontsize=8, ncol=3)
    save(fig, "kd_per_digit_gap")


def system_sheet_metrics(ctx):
    """Sheet exact-read rate and miscount-flag precision / recall for every model."""
    by = agg_rows(ctx)
    names = list(by)
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, (key, lab) in enumerate((("exact_read_rate", "sheet exact-read rate"), ("flag_precision", "flag precision"),
                                    ("flag_recall", "flag recall"))):
        ax.bar(np.arange(len(names)) + (k - 1) * 0.27, [np.mean([r[key] for r in by[n]]) for n in names], 0.27, label=lab)
    ax.set(xticks=range(len(names)), xticklabels=[n.replace(" <- ", "\n<- ").replace(" (", "\n(") for n in names],
           title="Does the system do its real job: read sheets and flag wrong totals?", ylim=(0, 1.05))
    ax.legend(fontsize=8)
    save(fig, "system_sheet_metrics")


def final_real_sheets(ctx):
    """Gallery of real photographed sheets (only after 04_final_test --final)."""
    f = OUT / "final_test.json"
    if not f.exists():
        return
    res = load_json(f)
    n = len(res["sheets"])
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4), squeeze=False)
    for ax, s in zip(axes[0], res["sheets"]):
        ax.axis("off")
        ax.set_title(f"{s['file']}\nread {s['read_marks']} / {s['read_total']}\n{'OK' if s['accepted'] else 'FLAG'}", fontsize=8,
                     color=C["acc"] if s["accepted"] else C["bad"])
    fig.suptitle(f"Real sheets: digit acc {res['digit_acc']:.2f} vs synthetic {res['synthetic_digit_acc']:.2f} (domain shift)",
                 fontweight="bold")
    save(fig, "final_real_sheets")


def main() -> None:
    """Load everything once and draw all figures."""
    ctx = {"index": load_json(ARCH / "index.json"), "v0": load_json(ARCH / "v000/full/metrics.json"),
           "best": load_json(ARCH / "best/full/metrics.json"), "v0conf": load_json(ARCH / "v000/full_conf/metrics.json"),
           "bestconf": load_json(ARCH / "best/full_conf/metrics.json"), "summary": load_json(OUT / "evolution_summary.json"),
           "kd": load_json(OUT / "distill_results.json")}
    for fn in (l0_accuracy_by_round, l0_accepted_rejected, l0_verifier_precision, l0_hard_examples,
               l0_verifier_vs_confidence, l0_confusion_delta, l1_evolution_timeline, l1_lineage_tree,
               l1_edit_effectiveness, l1_rejection_reasons, l1_best_vs_parent, l2_improver_utility,
               l2_improver_params, l2_ablation, kd_accuracy_vs_latency, kd_soft_targets, kd_tsweep_heatmap,
               kd_per_digit_gap, system_sheet_metrics, final_real_sheets):
        fn(ctx)


if __name__ == "__main__":
    main()
