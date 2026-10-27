# Real photographed sheets (final test, run ONCE)

1. Print `template/evomarks_template.pdf` (run `python scripts/00_make_template.py` first) on A4.
2. Fill 5-10 sheets **by hand** (you and friends; different pens and handwriting). Write one digit per box,
   the total as two digits (`07`, `23`). Make **1-2 sheets deliberately miscounted** (total not equal to the sum).
3. Photograph each sheet flat, in good light, **all four black corner squares visible**. Save as `sheet01.jpg`, ...
   into this folder.
4. Write `labels.json` here (see `labels.example.json`): what is *actually written* in each box,
   not what it should be. A miscounted sheet simply has `total` different from `sum(marks)`.
5. Only when everything else is finished: `python scripts/04_final_test.py --final`
   (also `python run_all.py --final`). It writes `results/FINAL_TEST_DONE` and warns loudly if you run it again.

Do not look at these results and then change the models: that would be tuning on the test set.
