# EvoMarks

A small research project: read a photographed exam mark sheet, recognise each handwritten digit, check that the marks add up
to the written total, and flag sheets that do not. It is small enough to run offline on a cheap phone, and it improves itself on three levels,
always under a **fixed, independent evaluator with rollback**.

> **Safety statement.** This is **bounded, sandboxed recursive self-improvement**. The course slides note that unrestricted recursive
> self-improvement has not been demonstrated; our system edits only two whitelisted files, under a fixed evaluator, with rollback.

| Level | What changes | Judged by |
|---|---|---|
| 0 · verified self-training | model weights | sum-check verifier + dev gate, rollback |
| 1 · self-modifying code | `pipeline.py` (preprocessing, augmentation, model, training settings) | fixed evaluator on `dev`, archive keeps every version |
| 2 · the recursive step | `improver.py` rewrites **its own** source | utility on a separate `meta_dev` split (STOP) |
| distillation | best teacher -> 25k-parameter student | accuracy, size, CPU latency, 3 seeds |

## Quick start
```
pip install -r requirements.txt
python run_all.py --quick          # ~3 min (demo sized, noisy)
python run_all.py                  # default run, ~12 min on a 12-core laptop
streamlit run app/streamlit_app.py # web app (read-only over archive/ and results/)
python demo.py photo.jpg           # read one photo (--version best|v000|student|vNNN)
python -m pytest tests -q          # 25 tests, a few seconds
```
`python run_all.py --clean` wipes `data/generated`, `archive`, `results` first. `--final` also runs the one-time real-sheet test
(see `data/real_sheets/README.md`; **not run yet**). Architecture: `docs/ARCHITECTURE.md`; written report: `docs/REPORT.md`.

## Results (default run, full mode, synthetic **test** split: 750 sheets, writers never seen in training)
Measured by `python run_all.py` (746 s total; `results/results_table.md`). Students: mean +- std over 3 seeds; teachers: one training each.

| model | digit acc | sheet exact-read (7/7) | flag precision | flag recall | params | size KB | latency ms/sheet (1 thread) |
|---|---|---|---|---|---|---|---|
| teacher v000, no loops | 0.8869 | 0.435 | 0.081 | 0.946 | 52,138 | 207 | 0.58 |
| teacher v000 + Level 0 | 0.9436 | 0.669 | 0.138 | 1.000 | 52,138 | 207 | 0.82 |
| teacher best (evolved v016) + Level 0 | 0.9455 | 0.683 | 0.143 | 1.000 | 81,019 | 320 | 0.87 |
| student hard-label | 0.9059 +- 0.0024 | 0.503 | 0.095 | 0.991 | 25,450 | 102 | 0.26 |
| student distilled from v000 | 0.9105 +- 0.0064 | 0.525 | 0.100 | 1.000 | 25,450 | 102 | 0.31 |
| student distilled from best | 0.9091 +- 0.0013 | 0.516 | 0.098 | 0.991 | 25,450 | 102 | 0.35 |

Honest reading (details and figures in `results/analysis_report.md`, standard error of a test accuracy is about 0.003):
- **Level 0 works**: the sum check lifts the seed-only teacher from 0.887 to 0.944 test accuracy (dev 0.891 -> 0.948), with pseudo-label precision 0.994.
  It beats a per-digit confidence threshold (precision 0.968, final dev 0.928 vs 0.948).
- **Level 1 found edits that raised dev accuracy at the small evolution budget** (v000 0.929 -> v016 0.958; meta_dev 0.943 -> 0.958), but
  **at full budget the evolved teacher is not clearly better than v000**: test 0.9455 vs 0.9436 is within noise, dev is identical (0.948). 18 of 20 candidates were rejected, all for "no improvement".
  Part of the dev gain is selection optimism and small-budget noise. Noise floor (`scripts/07_noise_floor.py`, 8 seeds): retraining v000 gives dev 0.934 +- 0.005, 2.4x the gate margin (0.002);
  paired by seed, v016 beats v000 by +0.0105 +- 0.0039 (SE) at the evolution budget (6/8 seeds), a small gain that disappears at full budget.
- **Level 2 produced a null result**: 3 self-edit proposals, 0 accepted; with no improver change, Level 1 + 2 and Level 1 only follow identical trajectories (3 seeds each, mean best dev 0.9538 for both), so that ablation says nothing about whether self-improvement could work. The utility is measured from one fixed parent, so edits of `PARENT_SELECTION` cannot change it: a limit of the gate, not a finding about recursion.
- **Distillation**: the student is 3.2x smaller (parameters) and 2.5-4x faster than the evolved teacher but 3.6 points less accurate; distilled vs hard-label is +0.3 points, **no clear effect**.
  Per-digit gaps hide a tail (digit 9: teacher is 21 points ahead).
- **Flagging**: recall is about 1.0 but precision only about 0.14, because most flags come from misreads, not genuine miscounts (the system says "check this sheet"; a human decides).

## Deviations from the original specification (and why)
- v000 uses `EPOCHS=8, LR=0.003` (spec example: 3 and 1e-3). With the spec values the seed-only teacher reached 36% digit accuracy, almost no sheet passed the sum check, and Level 0 could not start.
- `n_test = 750` sheets, not 1000: 1000 sheets need about 1300 images of the digit 1; MNIST test has 1135.
- `small_cnn` uses stride-2 convolutions instead of max-pooling (about 3.5x faster on the development CPU).
- The confidence baseline selects **per digit** (>= 0.9). A first version required all 7 digits on a sheet and accepted nothing, which would have been a strawman.
- Level 2 utility = mean `meta_dev` gain of children that the `dev` gate would accept (rejected edits count 0), same rng seeds for old and new improver.
- Evolved versions are written to `archive/`; `evolvable/*.py` stay as the seed (v000). `ast.unparse` drops comments (docstrings stay).
- Extra files not in the spec: the web app is split into `app/common.py` and `app/page_*.py`, to keep every file near 200 lines.
- Evolution used 3 seeds for the Level 1 vs Level 1+2 ablation; the main archive is seed 0. Distillation used 3 seeds. Teachers are single trainings.

## Limitations
Constrained edit space (AST operators over constants and whitelisted blocks); MNIST is cleaner than phone photos and **the real-photo test has not been run**;
single-digit marks and a fixed template; compensating errors pass the sum check (measured: 39 accepted sheets in the best run); many candidates selected on dev make dev optimistic;
small budgets and few seeds; latency measured on one machine, noisy at sub-millisecond scale; static checks plus a subprocess are not an OS-level sandbox.

## Layout
`config.py` all settings · `run_all.py` · `scripts/` steps 00-07 (07 = noise floor) · `evolvable/` the two editable seed files · `evomarks/` fixed library ·
`app/` Streamlit · `demo.py` · `tests/` · `docs/` (ARCHITECTURE, REPORT, slides).
Reproducibility: sheets are generated from code with fixed seeds; re-running the same settings reproduces the same numbers.
