# Slides outline (20 minutes: 3 problem and design, 5 method, 5 results and demo, 2 limits, rest Q&A)

| # | Slide | Time | Visual |
|---|---|---|---|
| 1 | **Title + the problem.** Handwritten marks and totals are typed by hand; totals are miscounted; we want an offline, phone-sized checker. | 1 min | `template/evomarks_template.png` + a rendered sheet (app **Try it** with a synthetic sheet, or `render_sheet` in `evomarks/sheets.py`) |
| 2 | **Idea in one picture.** Read digits, check the arithmetic, flag the sheet. Three improvement levels + distillation. | 2 min | app **Overview** diagram |
| 3 | **Level 0: verified self-training.** Sum-check verifier as independent judge; copy + dev gate + rollback. "A model agreeing with itself is not proof. Arithmetic is." | 2 min | `l0_accepted_rejected.png`, app Level 0 |
| 4 | **Level 1: code that rewrites code.** `pipeline.py`, AST operators, fixed evaluator, archive, rejected versions kept. | 2 min | `l1_lineage_tree.png`; show a diff |
| 5 | **Level 2: the recursive step.** improver edits itself; utility on separate meta_dev (STOP). Bounded, sandboxed. | 1 min | `l2_improver_utility.png` |
| 6 | **Guard-rails and data discipline.** Splits (writer-disjoint test), static check, tamper hashes, test read once, real sheets once. | 1 min | table from `docs/ARCHITECTURE.md`; app Safety page |
| 7 | **Results: Level 0 and Level 1.** v000 -> best, accepted vs rejected, which edits helped (and which never did). | 2 min | `l0_accuracy_by_round.png`, `l1_evolution_timeline.png`, `l1_edit_effectiveness.png` |
| 8 | **Results: ablations.** sum check vs confidence; Level 1 only vs Level 1 + 2 (3 seeds). Say plainly whether there is a difference. | 1.5 min | `l0_verifier_vs_confidence.png`, `l2_ablation.png` |
| 9 | **Distillation.** teacher vs hard-label vs distilled student; accuracy, size, latency, 3 seeds; the tail. | 1.5 min | `kd_accuracy_vs_latency.png`, `kd_per_digit_gap.png`, `kd_soft_targets.png` |
| 10 | **Failure analysis and limits.** compensating errors, domain shift on real photos, noise at small budgets, constrained edit space. | 2 min | `l0_hard_examples.png`, `final_real_sheets.png` (if run) |

## 2-minute live demo script
1. (0:00) `streamlit run app/streamlit_app.py` already open on **Try it**. "This is a photo of a sheet I filled by hand." Upload the photo.
2. (0:20) Point at the aligned sheet and the 7 crops. "Corner markers give a perspective warp, then each box is normalised like MNIST."
3. (0:35) Read the green tick or red warning. "Three models read it: v000, the evolved teacher, the 25k-parameter student. The check is arithmetic: marks sum to the total."
   If a model misreads, say so: the sum check usually exposes it (red warning), which is the point.
4. (1:00) Switch to **Version vs previous version**. Default is the best version: "Here is what the system changed in its own code": show the coloured diff, the metric deltas vs its parent, and the per-digit bars.
5. (1:30) Click "walk the lineage": v000 -> ... -> best, accuracy at each step; mention rejected siblings in the Level 1 page.
6. (1:50) End on the Safety page: "Two files editable, everything else hash-locked; and here are the limitations."

Backup if the photo fails: switch the radio to *Random synthetic test sheet* (same page, no upload needed).
