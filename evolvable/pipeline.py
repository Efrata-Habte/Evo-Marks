"""LEVEL 1 TARGET: the only source file that the evolution loop rewrites (besides improver.py).

Version 0 is DELIBERATELY MODEST: few epochs, no augmentation, only `normalize` as
preprocessing, a small CNN. That leaves room for the self-modifying loop to find gains.

Calibration note: with EPOCHS=3, LR=0.001 the seed-only teacher reached only 36% digit
accuracy, so almost no sheet passed the sum check and Level 0 could never bootstrap. EPOCHS=8,
LR=0.003 is a still-modest setting (no augmentation, tiny CNN) that does bootstrap.

The system may edit the constants below and the block calls inside preprocess/augment.
It can only use whitelisted blocks from evomarks/blocks.py (see evomarks/safety.py).
"""
from evomarks import blocks, models

CONV_CHANNELS = [8, 16]
HIDDEN = 64
DROPOUT = 0.2
LR = 0.003
EPOCHS = 8
BATCH = 128
ROT_DEG = 0
SHIFT_PX = 0
SELF_TRAIN_ROUNDS = 3
POOL_PER_ROUND = 600
MIN_DELTA = 0.002


def preprocess(img):
    """Applied to every digit before the model, in training and at inference."""
    img = blocks.normalize(img)
    return img


def augment(img, rng):
    """Applied to training digits only (a different random draw every epoch). Version 0: none."""
    return img


def build_teacher():
    """The model that Level 0 trains."""
    return models.small_cnn(CONV_CHANNELS, HIDDEN, DROPOUT)
