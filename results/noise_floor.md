Noise floor of the Level 1 gate (8 training seeds, evolution budget, dev n=300 sheets).

| version | dev_acc mean +- std | meta_acc mean +- std |
|---|---|---|
| v000 | 0.9340 +- 0.0047 | 0.9359 +- 0.0081 |
| v016 | 0.9445 +- 0.0076 | 0.9457 +- 0.0086 |

Gate margin min_delta = 0.002. Seed std of v000 dev_acc = 0.0047 (2.4x the margin).

Paired gain v016 - v000 (same seed): 
- dev_acc: +0.0105 +- 0.0039 (SE), better in 6/8 seeds
- meta_acc: +0.0098 +- 0.0047 (SE), better in 6/8 seeds
