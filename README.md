# EvoMarks

Offline checker for handwritten exam mark sheets. It reads 5 question marks and a 2-digit total, then flags sheets where the marks don't add up to the total.

The project combines both topics:

- **Track A, recursive learning.** A bounded improvement loop with an independent verifier (the arithmetic sum-check).
  - **Level 0:** verified self-training with a dev gate and rollback.
  - **Level 1:** the system edits its own `pipeline.py`. Every version is archived, accepted or rejected.
  - **Level 2:** the improver edits its own source and is kept only if it is a better improver.
- **Track B, knowledge distillation.** Teacher (v000 and the evolved best), a hard-label student and a distilled student.
- **Link between the tracks.** The evolved teacher's verified pseudo-labels become the student's training data.

## Setup

```
pip install -r requirements.txt
```

Python 3.10+ is needed. No GPU is required. MNIST is downloaded on the first run.

## Run

```
python run_all.py --quick     # ~5 min demo configuration
python run_all.py             # full configuration
python run_all.py --clean     # wipe data/generated, archive, results first
python -m pytest -q           # tests
streamlit run app/streamlit_app.py
```

`run_all.py` runs the scripts in order: `00` template, `01` generate sheets, `02` evolve, `03` distill, `05` figures, `06` analysis report.

Optional, **run once only**: `python run_all.py --final` also runs `04_final_test.py` on real photographed sheets. See [data/real_sheets/README.md](data/real_sheets/README.md).

## Data and reproducibility

- Sheets are generated from MNIST by code, with fixed seeds (`data_seed=123`, `seed=0`; KD seeds 0, 1, 2).
- All settings are in [config.py](config.py).
- Splits (full config, in sheets of 7 digits):

| Split | Sheets | Use |
|---|---|---|
| seed | 150 | labeled training data |
| pool | 3000 | unlabeled, for self-training |
| dev | 300 | Level 0 gate and Level 1 fitness |
| meta_dev | 300 | Level 2 gate only |
| test | 750 | reporting only, writer-disjoint, read once in `03_distill.py` |

## Outputs

Everything is written under `results/` and `archive/`, which are gitignored.

- `results/distill_results.json`: accuracy, size and latency per model and seed.
- `results/evolution_summary.json`: Level 1 and Level 2 histories and ablations.
- `results/*.png`: charts, from `05_make_figures.py`.
- `archive/`: every pipeline and improver version, with metrics and the accept or reject reason.

## Layout

```
evomarks/    core library (data, models, verifier, self-training, evaluator, safety)
evolvable/   the two seed files the system may rewrite: pipeline.py, improver.py
scripts/     numbered pipeline steps
app/         Streamlit explorer
tests/       pytest suite
docs/        ARCHITECTURE, CONCEPTS, WALKTHROUGH, SLIDES_OUTLINE
```

## Safety

Only `archive/` and `results/` are writable by evolved code. Evolved code is statically checked and runs in a sandboxed subprocess. Evaluator files are hash-checked for tampering. See `docs/ARCHITECTURE.md`.

## Limitations

- Sheets are composed from MNIST digits, so they are a proxy for real handwriting (domain shift is untested until the real-sheet test is run).
- Dev is small, so small accept thresholds can sit inside seed noise.
- Latency is measured on one machine with one thread.

See `docs/` for details and the failure analysis.
