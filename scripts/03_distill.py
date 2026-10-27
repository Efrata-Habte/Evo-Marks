"""Step 3: teachers vs students. Hard-label student, student distilled from teacher v000, student
distilled from the best evolved teacher; 3 seeds; T/alpha sweep on DEV; report on TEST (read here, once)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from config import ROOT, Config, get_config
from evomarks import blocks, models
from evomarks.archive import load_reader
from evomarks.data import load_split, sheets_to_digits
from evomarks.distillation import teacher_logits_for, to_input, train_student
from evomarks.metrics import confusion, count_params, cpu_latency_ms, model_size_kb, per_digit_acc, sheet_metrics
from evomarks.utils import save_json, set_seed

ARCH = ROOT / "archive"
OUT = ROOT / "results"


def transfer_set(best_dir: Path, seed_s, pool):
    """Student training data: labeled seed digits + the pseudo-labeled digits the best teacher's verified
    self-training kept. Identical for every student, so only the targets differ."""
    z = np.load(best_dir / "full" / "l0_final.npz")
    s_img, s_y = sheets_to_digits(seed_s)
    p_img = pool.images[z["sheet_idx"], z["box"]]
    return np.concatenate([s_img, p_img]), np.concatenate([s_y, z["labels"]])


def score(reader, split, cfg: Config, name: str, kind: str, seed: int) -> dict:
    """Test metrics of one model: accuracy, sheet-level scores, size and 1-thread CPU latency."""
    img, y = sheets_to_digits(split)
    pred = reader.probs(img).argmax(1)
    row = {"name": name, "kind": kind, "seed": seed, "params": count_params(reader.model),
           "size_kb": model_size_kb(reader.model), "latency_ms": cpu_latency_ms(reader, split.images, cfg.latency_runs),
           "per_digit_acc": per_digit_acc(y, pred), "confusion": confusion(y, pred).tolist()}
    row.update(sheet_metrics(pred.reshape(-1, 7), split.digits, split.is_miscounted))
    return row


def main() -> None:
    """Sweep, train students for each seed, evaluate everything on test, save results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    cfg = get_config(True if ap.parse_args().quick else None)
    torch.set_num_threads(1)
    best = ARCH / "best"
    teachers = {"v000": load_reader(ARCH / "v000", "v000", "full/model.pt"),
                "best": load_reader(best, "best", "full/model.pt")}
    noloop = load_reader(ARCH / "v000", "v000_noloop", "full/model_seed_only.pt")
    seed_s, pool, dev, test = (load_split(k) for k in ("seed", "pool", "dev", "test"))
    imgs, y_np = transfer_set(best, seed_s, pool)
    x, y = to_input(imgs), torch.from_numpy(y_np.astype(np.int64))
    dimg, dy = sheets_to_digits(dev)
    dx = to_input(dimg)
    t_logits = {k: teacher_logits_for(r.model, torch.from_numpy(np.stack([r.preprocess(i) for i in imgs])[:, None].astype(np.float32)))
                for k, r in teachers.items()}
    print(f"transfer set: {len(y)} digits; student params={count_params(models.tiny_student(cfg.student_hidden))}")

    def fit(tk, T, alpha, seed):
        """Train one student (tk=None: hard-label baseline) with the shared data and budget."""
        set_seed(seed)
        return train_student(models.tiny_student(cfg.student_hidden), x, y, t_logits.get(tk), T, alpha,
                             cfg.student_epochs, seed)

    sweep, chosen = {}, {}
    for tk in teachers:                      # choose T/alpha on DEV only (never on test)
        sweep[tk] = []
        for T in cfg.temperatures:
            for a in cfg.alphas:
                m = fit(tk, T, a, 0)
                dev_acc = float((m(dx).argmax(1).numpy() == dy).mean())
                sweep[tk].append({"T": T, "alpha": a, "dev_acc": dev_acc})
                print(f"  sweep {tk} T={T} alpha={a}: dev={dev_acc:.4f}")
        chosen[tk] = max(sweep[tk], key=lambda r: r["dev_acc"])
    rows = [score(noloop, test, cfg, "teacher v000 (no loops)", "teacher", 0),
            score(teachers["v000"], test, cfg, "teacher v000 (+L0)", "teacher", 0),
            score(teachers["best"], test, cfg, "teacher best (evolved)", "teacher", 0)]
    (OUT / "students").mkdir(parents=True, exist_ok=True)
    for seed in cfg.kd_seeds:
        for name, tk in (("student hard-label", None), ("student distilled <- v000", "v000"),
                         ("student distilled <- best", "best")):
            c = chosen.get(tk, {"T": 1.0, "alpha": 1.0})
            model = fit(tk, c["T"], c["alpha"], seed)
            rows.append(score(models.Reader(name, model, blocks.normalize), test, cfg, name, "student", seed))
            torch.save(model.state_dict(), OUT / "students" / f"{name.replace(' ', '_').replace('<-', 'from')}_s{seed}.pt")
            print(f"  {name} seed {seed}: test acc={rows[-1]['digit_acc']:.4f}")
    tx, ty = test.images.reshape(-1, 28, 28)[:700], test.digits.reshape(-1)[:700]   # soft-target explorer sample
    tl = teacher_logits_for(teachers["best"].model, torch.from_numpy(np.stack([teachers["best"].preprocess(i) for i in tx])[:, None].astype(np.float32)))
    np.savez_compressed(OUT / "kd_softtargets.npz", images=tx, labels=ty, logits=tl.numpy())
    save_json({"chosen": chosen, "sweep": sweep, "rows": rows, "transfer_size": len(y), "quick": cfg.quick,
               "student_hidden": cfg.student_hidden}, OUT / "distill_results.json")
    df = pd.DataFrame(rows).groupby("name")[["digit_acc", "exact_read_rate", "params", "size_kb", "latency_ms"]].mean()
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
