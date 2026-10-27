# Slide concept -> file / function

| Concept from the course | Where it lives | One-line explanation |
|---|---|---|
| **Independent verifier** | `evomarks/verifier.py::verify`, `verify_batch` | Arithmetic check (marks <= max, marks sum to total). It never sees confidence or labels. |
| **Independent evaluator** (never the model or improver grading itself) | `evomarks/evaluator.py::run` / `worker` | Fixed file, runs the candidate in a subprocess, scores on labeled splits the candidate cannot reach. |
| **Protected holdout** | `evomarks/data.py` (splits), `scripts/03_distill.py` (only reader of `test`), `scripts/04_final_test.py` | Test is read once, in one place; real sheets once, guarded by `--final` and `FINAL_TEST_DONE`. |
| **Accept / reject gate with rollback** | `self_training.py::run_level0` (comment `# Gate:`), `02_evolve.py::level1_step`, `level2_step` | Candidate replaces the parent only if it beats it by a margin; otherwise rolled back and logged with a reason. |
| **Archive / Darwin Goedel Machine** | `evomarks/archive.py::Archive`, `index.json` | Every version (also rejected ones) kept with source, parent, edit, metrics, reason. |
| **Constrained program search instead of an LLM** | `evomarks/code_editor.py` | AST parse -> edit -> `ast.unparse`; operators `tweak_constant`, `add_block`, `remove_block`, `resize_model`. |
| **STOP: the improver improves itself** | `evolvable/improver.py::propose_self_edit`, `02_evolve.py::level2_step`, `start_utility` / `collect_utility` | A new improver is kept only if its **utility** (mean meta_dev gain per edit from a fixed parent, same rng seeds) beats the old one by `min_delta_l2`. |
| **Separate gate data for the recursive step** | `meta_dev` split, `metrics["meta_acc"]` | Level 1 selects on dev; Level 2 is judged on meta_dev so an improver cannot win by overfitting dev. |
| **Reward-hacking guard** | `evomarks/safety.py::check`, `snapshot`, `assert_unchanged` | Forbidden names (`verifier`, `labels`, `test`, `open`, `exec`...), import whitelist, SHA-256 tamper check around every evaluation. |
| **Sandboxed execution** | `safety.py::run_sandboxed`, `evaluator.run` | Subprocess + timeout; crashes and timeouts become rejected versions with a reason. |
| **Model-collapse / diversity guard** | `self_training.py` (`max_class_drop`, seed data always mixed in, data committed only on accepted rounds), tournament parent selection in `improver.py` | A candidate that sacrifices a digit class is rejected; pseudo-labels never replace real labels; parents are not always the single best. (No explicit diversity metric: stated limitation.) |
| **Pseudo-labelling / self-training** | `self_training.py::select_digits`, `run_level0` | Accepted sheets' read digits become labels for a copy of the teacher. |
| **Confirmation bias baseline** | `select_digits(mode="confidence")`, `02_evolve.py::finalize` (`full_conf` runs) | Per-digit confidence >= 0.9: "a model agreeing with itself". |
| **Compensating errors (verifier weakness)** | `self_training.py::_audit` (`compensating_errors`) | Two misreads that still sum correctly pass the check; counted afterwards with hidden labels. |
| **Soft targets, temperature T** | `evomarks/distillation.py::distillation_loss`, figure `kd_soft_targets` | `softmax(teacher / T)`; T > 1 exposes dark knowledge. |
| **T^2 factor** | `distillation_loss` (`* T * T`) | Restores gradient magnitude after dividing logits by T. |
| **alpha, hard-label baseline** | `distillation_loss`, `03_distill.py` (`student hard-label`) | `alpha = 1` is plain cross-entropy = no distillation (tested). |
| **Capacity gap / averages hide the tail** | `03_distill.py`, figure `kd_per_digit_gap` | Tiny student vs teacher per digit, accuracy vs latency vs size. |
| **Reproducibility** | `evomarks/utils.py::set_seed`, `config.py`, fixed `data_seed` | Sheets are generated from code; identical settings give identical numbers (verified). |
| **Bounded recursive self-improvement** | README safety statement, `docs/ARCHITECTURE.md` | Two whitelisted files, fixed evaluator, rollback. Not open-ended self-improvement. |
