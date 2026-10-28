# EvoMarks: verified self-training, self-modifying code and distillation for reading handwritten mark sheets

## Problem
Exam cover sheets hold handwritten marks per question and a handwritten total that people type in by hand; miscounted totals and typos are common.
We want a small offline system that reads a photographed sheet, recognises each digit, checks that marks sum to the total and flags sheets that do not.

## Method
**Sheets.** 5 questions (maxima 5, 8, 9, 6, 7), one digit each, plus a 2-digit total: 7 digit boxes. Synthetic sheets are built from MNIST digits;
about 5% of sheets get a genuinely wrong total. `seed` (150 sheets, labeled), `pool` (3000, unlabeled), `dev` and `meta_dev` (300 each) come from MNIST train; `test` (750) from MNIST test, written by other people. No image is in two splits.
**Level 0, verified self-training.** The teacher reads pool sheets; a sheet's digits become pseudo-labels only if every mark is within its maximum and the marks sum to the read total.
Accepted sheets train a copy of the teacher, adopted only if dev accuracy improves by at least 0.002 and no digit class loses more than 5 points; otherwise rollback.
**Level 1, self-modifying code.** An improver edits `pipeline.py` (constants, preprocessing and augmentation blocks, model size) with AST operators. A fixed evaluator runs each candidate in a subprocess with a timeout under one small budget (2 rounds of 400 pool sheets) and scores dev accuracy; versions that improve by 0.002 enter the archive, the rest are rejected and logged. A static check and SHA-256 tamper hashes protect the evaluator, data and tests.
**Level 2, the recursive step.** Every 5 generations the improver proposes an edit of its own source. It is kept only if its utility (mean meta_dev gain per proposed edit from a fixed parent, same random seeds) beats the old improver's by 0.001.
**Distillation.** `L = alpha CE + (1-alpha) T^2 KL`. A 25k-parameter MLP student is trained on the same data (seed digits plus the best teacher's accepted pseudo-labels) as hard-label baseline and distilled from v000 and from the best teacher; T and alpha chosen on dev, 3 seeds, report on test.

## Setup
CPU only, 12 cores; one thread for latency (median of 200 runs after warm-up, preprocessing included). Default run 746 s; `--quick` run 168 s. Seeds fixed; identical settings reproduce identical numbers (checked).

## Results (test split, 5,250 digits; standard error of an accuracy about 0.003)
| model | digit acc | sheet exact-read | flag precision / recall | params | latency ms |
|---|---|---|---|---|---|
| v000, no loops | 0.887 | 0.435 | 0.08 / 0.95 | 52,138 | 0.58 |
| v000 + Level 0 | 0.944 | 0.669 | 0.14 / 1.00 | 52,138 | 0.82 |
| best evolved (v016) + Level 0 | 0.946 | 0.683 | 0.14 / 1.00 | 81,019 | 0.87 |
| student, hard-label | 0.906 +- 0.002 | 0.503 | 0.10 / 0.99 | 25,450 | 0.26 |
| student, distilled from v000 | 0.911 +- 0.006 | 0.525 | 0.10 / 1.00 | 25,450 | 0.31 |
| student, distilled from best | 0.909 +- 0.001 | 0.516 | 0.10 / 0.99 | 25,450 | 0.35 |

- Level 0: dev accuracy 0.891 -> 0.948 for v000; pseudo-label precision 0.994 vs 0.968 for a per-digit confidence threshold (>= 0.9), final dev 0.948 vs 0.928. The sum check rescued 188 low-confidence digits (175 correct) in the v000 run.
- Level 1: 20 generations, 2 accepted (v011 channels [8,25]; v016 random_shift), 18 rejected (all `no_improvement`; 0 unsafe, crashed or timed out). Evolution-budget dev 0.929 -> 0.958, meta_dev 0.943 -> 0.958. At full budget v000 and v016 reach the same dev accuracy (0.948); test 0.9455 vs 0.9436 is within noise.
- Level 2: 3 proposals, 0 accepted (utilities +0.000/+0.000, -0.004/-0.004, +0.004/+0.004). Level 1 + 2 and Level 1 only therefore coincide (mean best dev 0.9538 over 3 seeds each; per seed 0.958, 0.946, 0.957).
- Distillation: 3.2x fewer parameters than the evolved teacher, about 2.5-4x lower latency, 3.6 points lower accuracy; distilled vs hard-label +0.3 points (not distinguishable from seed noise).
- Safety: 184 tamper checks, 0 violations.

## Failure analysis
- **Edits that never helped**: 9 of 10 `add_block` edits, the single `remove_block` edit, all 6 constant tweaks and 2 of 3 model resizes were rejected; on average each operator type lowered dev accuracy versus its parent (mean change -0.014, -0.023, -0.005, -0.020). Evolution is mostly rejection.
- **Noise floor of the gate** (`scripts/07_noise_floor.py`, 8 training seeds, evolution budget): retraining v000 gives dev accuracy 0.934 +- 0.005, i.e. a seed std 2.4x the acceptance margin (0.002). A single evaluation can therefore pass or fail a candidate by chance, and with 20 candidates the best dev score is optimistic. Paired by seed, v016 beats v000 by +0.0105 +- 0.0039 (SE) on dev and +0.0098 +- 0.0047 on meta_dev, better in 6 of 8 seeds: a small but probably real gain at the evolution budget.
- **Evolution gain did not transfer**: the evolution-budget gain (+1 point paired, +3 points as first selected) vanishes at full budget: v000 and v016 reach the same dev accuracy (0.948) and test 0.9455 vs 0.9436 is within noise. The loop optimised the small budget, not the final system.
- **Level 2 cannot accept some edits by construction**: utility is measured from one fixed parent, so an edit of `PARENT_SELECTION` cannot change it (gen 5 and 10 gave identical old/new utility). With 0 accepted self-edits the improver never changes, so "Level 1 + 2 vs Level 1 only" compares identical processes (difference exactly +0.0000): a null result about the gate and metric, not evidence that self-improvement of the improver is impossible.
- **Compensating errors**: 19 accepted sheets (v000 run) and 39 (best run) had wrong digits that still summed correctly; precision stays high (0.994 / 0.986) but is not 1.
- **Flag precision is low (0.14)** at recall about 1.0: most flagged sheets are misreads, not genuine miscounts; a flag means "check this sheet".
- **Student tail**: digit 9 is 21 points worse than the teacher; averages hide this.
- **Real-photo domain shift**: not yet measured; the real-sheet test is prepared (`04_final_test.py --final`) but has not been run. A synthetic perspective-distorted photo was read identically to the original digits, which only validates the photo pipeline, not real handwriting.

## How the two tracks connect (interfaces and trade-offs)
- **Recursive learning -> distillation.** The only interface is data: the best teacher's verified pseudo-labels (`archive/best/full/l0_final.npz`) plus the labeled seed digits form the student's transfer set; the teacher's logits supply the soft targets. Every student sees the same set, so only the targets differ.
- **Distillation -> recursive learning (not built).** A student could be the cheap proposer or scorer inside the loop, or be re-distilled after each accepted teacher update; this would make each generation cheaper but couples the loop to a weaker model.
- **Trade-offs.** Pseudo-label noise (precision 0.986-0.994, compensating errors) passes straight into the student. The loop costs 20 evaluations plus Level 0 rounds, while the student gives 3.2x fewer parameters and about 2.5-4x lower latency at 3.6 points of accuracy. The verified-feedback improvement of the teacher did not carry over: distilling from the evolved teacher was not better than from v000.

## Limitations
Constrained edit space (constants and whitelisted blocks, not a code-writing model); MNIST is cleaner than phone photos; single-digit marks and a fixed template; dev optimism after selecting among many versions; small budgets and a handful of seeds (teachers are single trainings); latency from one machine; static checks and a subprocess are not an OS-level sandbox; Level 2 only changes a few numeric settings and one selection rule.
