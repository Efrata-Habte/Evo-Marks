# Read-the-repo-in-one-day guide

About 8 hours with breaks. Every step: what to read, what to run, and the questions an evaluator might ask (with short answers).
Run commands from the repository root. Setup once: `pip install -r requirements.txt`, then `python run_all.py --quick` (about 3-4 minutes)
so that `archive/` and `results/` exist.

## Step 1 · Big picture (30 min)
Read `README.md` and `docs/ARCHITECTURE.md`. Be able to say in one breath: *Level 0 changes weights, Level 1 changes code,
Level 2 changes the code that changes code, and a fixed evaluator with rollback judges all of it. Then distillation shrinks the winner.*

**Questions**
- *Is this real recursive self-improvement?* No, and we say so: it is **bounded, sandboxed** recursion. Two whitelisted files, edits are AST operators over
  constants and whitelisted blocks, a fixed evaluator, rollback. Unrestricted recursive self-improvement has not been demonstrated.
- *What is the real-world job?* Flag sheets whose written total is wrong, offline, on a cheap phone.

## Step 2 · Sheets and the verifier (1 h)
Read `config.py` (all settings), `evomarks/sheets.py`, `evomarks/verifier.py`, `evomarks/data.py`.
Demo (prints one accepted and two rejected sheets):
```
python -c "from evomarks.verifier import verify; from config import MAXIMA; print(verify([1,2,3,4,5],15,MAXIMA)); print(verify([1,2,3,4,5],16,MAXIMA)); print(verify([6,0,0,0,0],6,MAXIMA))"
```
Then look at `results/p0_*.png` if present, or render one: `python -c "from evomarks.data import load_split; from evomarks.sheets import render_sheet; render_sheet(load_split('dev'), 0).save('sheet.png')"`.

**Questions**
- *Why split by writer?* MNIST train and test were written by different people; building `test` from MNIST test means the final synthetic score is on unseen writers, so a model cannot win by memorising pen styles. Each image is used in one split only.
- *How could the sum check be fooled?* **Compensating errors**: two misreads that still sum right (a 3 read as 5 and a 5 as 3), or a systematic confusion that is consistent across boxes. We count them afterwards with hidden labels (`_audit`). A second weakness: genuinely miscounted sheets are *correctly* rejected but their (right) digits are wasted. A third: accepted sheets are the "easy" ones, so pseudo-labels are biased towards clear handwriting.
- *Why is `miscount_rate` 5%?* Real humans make arithmetic slips; it lets us measure the real job (flag precision / recall).

## Step 3 · Level 0 and its round log (1 h)
Read `evomarks/self_training.py` top to bottom (`run_level0` is the algorithm; `_audit` is measurement only).
Run: `python -c "import json; [print({k:v for k,v in r.items() if k!='per_digit_acc'}) for r in json.load(open('archive/best/full/metrics.json'))['rounds']]"`

**Questions**
- *Why a model copy and a dev gate?* Self-training can confirm its own mistakes. The copy is only adopted if dev accuracy improves by `MIN_DELTA` and no class drops more than 5 points.
- *Why is the verifier better than a confidence threshold?* Confidence measures agreement with oneself; arithmetic is external evidence. See the `confidence >= 0.9` ablation (`full_conf` runs, figure `l0_verifier_vs_confidence`) and the count of **low-confidence digits rescued** (`low_conf_accepted`).
- *Where are the hidden labels used?* Only in `evaluate` (scoring on dev/meta_dev) and `_audit`; the loop's decisions use images and the verifier only.

## Step 4 · Level 1: code that rewrites code (1.5 h)
Read in this order: `evolvable/pipeline.py`, `evomarks/blocks.py`, `evomarks/code_editor.py`, `evomarks/evaluator.py`, `scripts/02_evolve.py` (`level1_step`).
Run a sample of edits:
```
python -c "import random; from evomarks import code_editor as ce; s=open('evolvable/pipeline.py').read(); r=random.Random(1); [print(ce.apply_operator(o,s,r,0.3)[1]) for o in ce.OPERATORS]"
```
Look at `archive/index.json` and at a diff: `python -c "import difflib;a=open('archive/v000/pipeline.py').read().splitlines();b=open('archive/best/pipeline.py').read().splitlines();print(chr(10).join(difflib.unified_diff(a,b,'v000','best',lineterm='')))"`.

**Questions**
- *Why can't the system edit the evaluator?* If it could, the cheapest "improvement" is to change the score, not the model (reward hacking). `safety.check` bans references to `verifier`, `evaluator`, `labels`, `test`..., and every evaluation is wrapped in SHA-256 hashes of `evomarks/`, `data/generated/`, `tests/`, `config.py`.
- *Why no LLM?* Cost, reproducibility and explainability: AST operators are a small constrained version of what DGM does with an LLM, and every candidate is syntactically valid by construction.
- *Why a subprocess?* Isolation and a hard timeout; crashes become rejected versions with a reason, not a dead loop.
- *Why does `ast.unparse` matter?* It guarantees valid code but drops comments (docstrings stay); v000 is normalised the same way so diffs only show real edits.
- *Is the dev score of the best version trustworthy?* No, it is optimistic: it was *selected* on dev among many candidates. We therefore also report `meta_acc` and, later, test.

## Step 5 · Level 2 and the guard-rails (1 h)
Read `evolvable/improver.py`, `02_evolve.py` (`level2_step`, `start_utility`, `collect_utility`), `evomarks/safety.py`, `evomarks/archive.py`.
Run `python -c "from evomarks import safety; print(safety.check('import os'), safety.check(open('evolvable/pipeline.py').read()))"`.

**Questions**
- *What exactly is the recursive step?* Every `improver_every` generations the improver proposes an edit of **its own source** (operator weights, step size, parent selection). Both the old and new improver propose `utility_edits` children from the same fixed parent with the same rng seeds; utility = mean meta_dev gain of the children the dev gate would accept (rejected edits count 0). The new improver is kept only if `u_new > u_old + min_delta_l2`.
- *Why a separate meta_dev?* Level 1 already used dev for selection. If the improver were also judged on dev, it could win by overfitting dev. meta_dev is only touched by the Level 2 gate.
- *What are the limits of the tamper check?* It catches modification of protected files during an evaluation; it is not an OS sandbox (a determined candidate with arbitrary code could do more), which is why the static whitelist forbids `os`, `open`, `exec`...

## Step 6 · Distillation (1 h)
Read `evomarks/distillation.py` (about 50 lines) and `scripts/03_distill.py`.
Run `python -c "import json; d=json.load(open('results/distill_results.json')); print(d['chosen']); [print(r['name'], r['seed'], round(r['digit_acc'],4)) for r in d['rows']]"`.

**Questions**
- *What does alpha = 1 do?* Removes the KL term: plain cross-entropy, i.e. no distillation (the hard-label baseline). Tested in `tests/test_distill_photo.py`.
- *Why does T > 1 help?* Dividing logits by T flattens the teacher's distribution so the small probabilities of the wrong-but-similar classes (a 4 that looks like a 9) become visible training signal ("dark knowledge"). See `kd_soft_targets`.
- *Why multiply by T^2?* Soft-target gradients shrink by 1/T^2; the factor keeps their magnitude comparable to the CE term.
- *How is the comparison fair?* Same student, same data (seed digits + the best teacher's kept pseudo-labels), same epochs, 3 seeds; T/alpha chosen on **dev**, reported on **test**.
- *Did distillation help?* Read the numbers in `results/analysis_report.md`; the verdict there uses a 2-standard-error rule and says "no clear effect" when that is the honest answer.

## Step 7 · Figures, report, web app (1 h)
Read `scripts/05_make_figures.py` (one function per figure) and `scripts/06_analysis_report.py` (sentences are computed from numbers). Then `streamlit run app/streamlit_app.py` and click through: Overview, Try it, **Version vs previous version**, Distillation, Level 1, Level 0, Level 2, Safety.

**Questions**
- *How do I know the numbers are not cherry-picked?* Fixed seeds, sheets generated from code, rejected versions kept in the archive, examples galleries show the *first* items found, and the report states "no clear effect" when differences are within noise.
- *Why is latency per sheet measured with 1 thread?* To mimic a single phone core; median of 200 runs after warm-up, preprocessing included. It is one machine: a stated limitation.

## Step 8 · Slides (last hour)
Open `docs/SLIDES_OUTLINE.md`, then rehearse the 2-minute demo with the app.
