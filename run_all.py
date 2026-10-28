"""Run the whole project: python run_all.py [--quick] [--clean] [--final]
Steps 00,01,02,03,05,06 always; 04 (real photographed sheets, runs ONCE) only with --final."""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["00_make_template", "01_generate_sheets", "02_evolve", "03_distill", "07_noise_floor", "05_make_figures", "06_analysis_report"]
QUICK_AWARE = {"01_generate_sheets", "02_evolve", "03_distill", "07_noise_floor"}


def main() -> None:
    """Run each script as its own process so a failure stops the pipeline with a clear message."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="~5 minute demo configuration")
    ap.add_argument("--clean", action="store_true", help="delete data/generated, archive and results first")
    ap.add_argument("--final", action="store_true", help="also run the one-time real-sheet test (04)")
    args = ap.parse_args()
    if args.clean:
        for d in ("data/generated", "archive", "results"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
    steps = STEPS + (["04_final_test"] if args.final else [])
    t_all, times = time.time(), {}
    for step in steps:
        cmd = [sys.executable, str(ROOT / "scripts" / f"{step}.py")]
        cmd += ["--quick"] if args.quick and step in QUICK_AWARE else []
        cmd += ["--final"] if step == "04_final_test" else []
        print(f"\n=== {step} ===", flush=True)
        t0 = time.time()
        if subprocess.run(cmd, cwd=ROOT).returncode:
            sys.exit(f"{step} failed")
        times[step] = time.time() - t0
    print("\n" + "\n".join(f"{k:22s}{v:7.1f}s" for k, v in times.items()) + f"\n{'total':22s}{time.time() - t_all:7.1f}s")
    print("Next: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
