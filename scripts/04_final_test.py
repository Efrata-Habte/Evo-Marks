"""Step 4: the FINAL test on real photographed sheets. Runs ONCE, only with --final.

Nothing here feeds back into training or selection: the models are already frozen. Reading the real
sheets more than once would let us tune on them, so a re-run prints a loud warning."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import MAXIMA, ROOT
from evomarks.archive import default_readers
from evomarks.photo import read_photo
from evomarks.utils import load_json, save_json
from evomarks.verifier import verify

REAL = ROOT / "data" / "real_sheets"
DONE = ROOT / "results" / "FINAL_TEST_DONE"


def main() -> None:
    """Read every labeled real sheet with v000 / best / student and report domain shift."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    if not ap.parse_args().final:
        sys.exit("Refusing to run: the real-sheet test must be run once, deliberately. Pass --final.")
    if DONE.exists():
        print("!" * 70 + f"\nWARNING: the final test was already run ({DONE.read_text().strip()}).\n"
              "Re-running and then changing anything would be tuning on the test set.\n" + "!" * 70)
    labels = load_json(REAL / "labels.json")["sheets"]
    readers = default_readers(ROOT / "archive", ROOT / "results")
    synthetic = {r["name"]: r["digit_acc"] for r in load_json(ROOT / "results" / "distill_results.json")["rows"]}
    out = {"models": {}, "sheets": []}
    for name, reader in readers.items():
        right = total = exact = flags_ok = 0
        for item in labels:
            res = read_photo(REAL / item["file"])
            truth = np.array(item["marks"] + [item["total"] // 10, item["total"] % 10])
            pred = reader.probs(res.crops).argmax(1)
            v = verify([int(x) for x in pred[:5]], int(pred[5]) * 10 + int(pred[6]), MAXIMA)
            miscounted = item["total"] != sum(item["marks"])
            right, total, exact = right + int((pred == truth).sum()), total + 7, exact + int((pred == truth).all())
            flags_ok += int((not v.accepted) == miscounted)
            if name == "best":
                out["sheets"].append({"file": item["file"], "read_marks": pred[:5].tolist(), "read_total": int(pred[5]) * 10 + int(pred[6]),
                                      "accepted": v.accepted, "miscounted": miscounted, "warnings": res.warnings})
        out["models"][name] = {"digit_acc": right / total, "exact_read_rate": exact / len(labels),
                               "flag_correct_rate": flags_ok / len(labels)}
        print(f"{name:8s} real digit acc {right / total:.3f}  exact {exact / len(labels):.3f}  flag correct {flags_ok / len(labels):.3f}")
    out["digit_acc"] = out["models"]["best"]["digit_acc"]
    out["synthetic_digit_acc"] = synthetic.get("teacher best (evolved)", float("nan"))
    save_json(out, ROOT / "results" / "final_test.json")
    DONE.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"synthetic test acc {out['synthetic_digit_acc']:.3f} vs real {out['digit_acc']:.3f}; wrote results/final_test.json")


if __name__ == "__main__":
    main()
