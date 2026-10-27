"""Verifier, sheet generator, split disjointness and the photo reader."""
import numpy as np

from config import MAXIMA
from evomarks.photo import read_photo
from evomarks.sheets import render_sheet
from evomarks.verifier import verify


def test_verifier_accepts_correct_sheet():
    assert verify([1, 2, 3, 4, 5], 15, MAXIMA).accepted


def test_verifier_rejects_sum_mismatch():
    r = verify([1, 2, 3, 4, 5], 16, MAXIMA)
    assert not r.accepted and r.reason == "sum_mismatch"


def test_verifier_rejects_mark_over_max():
    r = verify([6, 0, 0, 0, 0], 6, MAXIMA)          # Q1 max is 5
    assert not r.accepted and r.reason == "mark_exceeds_max"


def test_generator_valid_marks_and_miscount_flag(tiny_splits):
    s = tiny_splits["pool"]
    assert (s.true_marks <= np.array(MAXIMA)).all() and (s.true_marks >= 0).all()
    wrong = s.written_total != s.true_marks.sum(axis=1)
    assert (wrong == s.is_miscounted).all()                 # flag matches reality
    assert 0 < s.is_miscounted.sum() < len(s)               # some, not all, are miscounted
    assert (s.written_total >= 0).all() and (s.written_total <= 35).all()


def test_splits_share_no_mnist_image(tiny_splits):
    train_like = ["seed", "pool", "dev", "meta_dev"]
    ids = [set(map(int, tiny_splits[k].mnist_idx.ravel())) for k in train_like]
    assert sum(len(i) for i in ids) == len(set().union(*ids))   # no overlap between train-derived splits
    for k in train_like:
        assert len(ids[train_like.index(k)]) == tiny_splits[k].mnist_idx.size   # no reuse inside


def test_images_match_labels_source(tiny_splits):
    from evomarks.data import load_mnist
    mn = load_mnist()
    s = tiny_splits["test"]                                  # test comes only from MNIST test
    assert (mn["test"][1][s.mnist_idx] == s.digits).all()
    d = tiny_splits["dev"]
    assert (mn["train"][1][d.mnist_idx] == d.digits).all()


def test_read_photo_returns_seven_28x28_crops(tiny_splits):
    img = render_sheet(tiny_splits["dev"], 0)
    res = read_photo(img)
    assert res.crops.shape == (7, 28, 28) and res.found_markers
