# EvoMarks: verified self-improvement and distillation for reading handwritten mark sheets

## 1. Problem and goal
Exam cover sheets hold handwritten marks per question and a handwritten total, and people type these in by hand; miscounted totals and typos are common. EvoMarks is a small offline system that reads a sheet, recognises each digit, checks that the marks sum to the total, and flags sheets that do not.
It covers both tracks: **Track A** (a bounded recursive-improvement loop with an independent verifier, Levels 0-2) and **Track B** (a teacher, a hard-label student and a distilled student).

## 2. Method
**Data.** A sheet has 5 marks (maxima 5, 8, 9, 6, 7) and a 2-digit total: 7 digit boxes. Sheets are built from MNIST digits with fixed seeds; about 5% get a genuinely wrong total. Splits (sheets): `seed` 150 (labeled), `pool` 3000 (unlabeled), `dev` 300 (gates), `meta_dev` 300 (Level 2 gate only), `test` 750 (MNIST-test writers, read once, reporting only). No image is in two splits.

**Level 0, verified self-training.** The teacher reads pool sheets. A sheet's digits become pseudo-labels only if each mark is within its maximum and the marks sum to the read total. A copy trained on them replaces the teacher only if dev accuracy rises by 0.002 and no digit class drops more than 5 points; otherwise rollback.
**Level 1, self-modifying code.** An improver edits `pipeline.py` (constants, preprocessing/augmentation blocks, model size) with AST operators. A fixed evaluator runs each candidate in a sandboxed subprocess under one small budget and scores dev accuracy; every version is archived, accepted or rejected, with its reason. Static checks and SHA-256 tamper hashes protect the evaluator, data and tests (184 checks, 0 violations).
**Level 2, the recursive step.** Every 5 generations the improver edits its own source; it is kept only if its utility (mean meta_dev gain of its proposed edits from a fixed parent, same random seeds) beats the old improver by 0.001.
**Track B.** Loss `alpha CE + (1-alpha) T^2 KL`. A 25k-parameter MLP student is trained on the same data (seed digits plus the best teacher's accepted pseudo-labels) as a hard-label baseline and distilled from v000 and from the best teacher. T and alpha are chosen on dev; 3 seeds; reported on test.

**Reproducibility.** CPU only (12 cores); latency is one thread, median of 200 runs, preprocessing included. `python run_all.py` rebuilds everything (evolution about 11.5 min); a re-run reproduced every accuracy exactly (only latency varies).

## 3. Results (test split, 5,250 digits; standard error of an accuracy about 0.003)

| model | digit acc | sheet exact-read | flag precision / recall | params | latency ms |
|---|---|---|---|---|---|
| teacher v000, no loops | 0.887 | 0.435 | 0.08 / 0.95 | 52,138 | 0.58 |
| teacher v000 + Level 0 | 0.944 | 0.669 | 0.14 / 1.00 | 52,138 | 0.82 |
| teacher best (v016) + Level 0 | 0.946 | 0.683 | 0.14 / 1.00 | 81,019 | 0.87 |
| student, hard-label | 0.906 +- 0.002 | 0.503 | 0.10 / 0.99 | 25,450 | 0.26 |
| student, distilled from v000 | 0.911 +- 0.006 | 0.525 | 0.10 / 1.00 | 25,450 | 0.31 |
| student, distilled from best | 0.909 +- 0.001 | 0.516 | 0.10 / 0.99 | 25,450 | 0.35 |

Teachers: one training each; students: mean +- std over 3 seeds. Figures are in `results/figures/`.

- **Level 0 works.** Seed-only teacher 0.887 to 0.944 test accuracy. Pseudo-label precision 0.994 for the sum check vs 0.968 for a per-digit confidence threshold (>= 0.9); final dev 0.948 vs 0.928.
- **Level 1: a small gain that does not survive full budget.** 20 candidates, 2 accepted (v011 `CONV_CHANNELS [8,25]`, v016 `random_shift`), 18 rejected (all `no_improvement`). Dev 0.929 to 0.958 at the evolution budget, but at full budget v000 and v016 both reach dev 0.948 and test 0.9455 vs 0.9436 is within noise.
- **Level 2: null result.** 3 self-edits, 0 accepted; Level 1 + 2 and Level 1 only both reach mean best dev 0.9538 (3 seeds).
- **Distillation.** The student has 3.2x fewer parameters and runs 2.5-4x faster than the evolved teacher at 3.6 points lower accuracy; distilled vs hard-label is +0.3 points, not distinguishable from seed noise.

## 4. Failure analysis
- **Noise floor of the gate** (8 seeds, evolution budget). Retraining v000 gives dev accuracy 0.934 +- 0.005, 2.4x the acceptance margin (0.002): one evaluation can pass or fail a candidate by chance, and the best of 20 candidates is optimistic. Paired by seed, v016 beats v000 by +0.0105 +- 0.0039 (SE) on dev and +0.0098 +- 0.0047 on meta_dev (6 of 8 seeds): small, probably real, at the evolution budget only. The loop optimised the cheap evaluator, not the final system.
- **Edits mostly hurt.** On average every operator type lowered dev accuracy versus its parent (add_block -0.014, remove_block -0.023, tweak_constant -0.005, resize_model -0.020). Evolution is mostly rejection.
- **Level 2 cannot accept some edits by construction.** Utility is measured from one fixed parent, so an edit of `PARENT_SELECTION` cannot change it (identical old and new utility at generations 5 and 10). With no accepted edit the improver never changes, so "Level 1 + 2 vs Level 1 only" compares identical processes (difference exactly 0). This is a limit of the gate, not evidence that self-improvement of the improver is impossible.
- **Compensating errors.** Wrong digits that still sum correctly passed the check (19 sheets in the v000 run, 39 in the best run); precision is high but not 1.
- **Low flag precision (0.14) at recall about 1.0.** Most flags are misreads, not genuine miscounts; a flag means "a human should check this sheet".
- **Student tail.** Digit 9 is 21 points worse than the teacher; averages hide this.
- **Real handwriting is untested.** The real-photo test was not run, so the synthetic test split is the final test. MNIST digits are cleaner than phone photos, so absolute accuracies are optimistic for real sheets.

## 5. How the two tracks connect
- **Track A to Track B (built).** The only interface is data: the best teacher's verified pseudo-labels plus the labeled seed digits form the student's transfer set, and the teacher's logits give the soft targets. All students see the same set, so only the targets differ.
- **Track B to Track A (not built).** A student could act as a cheap proposer or scorer inside the loop, or be re-distilled after each accepted teacher update. That would cheapen generations but tie the loop to a weaker model.
- **Trade-offs.** Pseudo-label noise (precision 0.986-0.994, compensating errors) passes straight into the student. The loop costs 20 evaluations plus Level 0 rounds, while the student saves 3.2x parameters and 2.5-4x latency for 3.6 points of accuracy. The teacher improvement from evolution did not carry over: distilling from the evolved teacher was not better than from v000.

## 6. Limitations
Edit space limited to constants and whitelisted blocks (no code-writing model); synthetic MNIST-based sheets and a fixed single-digit template; few seeds and single-training teachers; dev optimism after selecting among many candidates; latency from one machine; static checks plus a subprocess are not an OS-level sandbox; Level 2 only adjusts a few numeric settings and one selection rule. This is bounded, sandboxed recursion, not open-ended self-improvement.
