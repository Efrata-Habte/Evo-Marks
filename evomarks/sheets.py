"""Mark sheets: the Sheet dataclass, the synthetic generator and the sheet renderer.

Template: 5 questions (one handwritten digit each, printed maxima MAXIMA) plus a 2-digit
total (00-35) = 7 digit boxes. The box layout below is shared by the printable template,
`render_sheet` and `photo.read_photo` so all three always agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import MAXIMA, N_QUESTIONS

# ---- template geometry (A4 at 150 dpi); box order = Q1..Q5, total tens, total ones ----
W, H = 1240, 1754
MARKER, MARGIN, BOX = 90, 60, 160
CORNER_CENTERS = [(MARGIN + MARKER // 2, MARGIN + MARKER // 2),
                  (W - MARGIN - MARKER // 2, MARGIN + MARKER // 2),
                  (MARGIN + MARKER // 2, H - MARGIN - MARKER // 2),
                  (W - MARGIN - MARKER // 2, H - MARGIN - MARKER // 2)]   # TL TR BL BR
BOXES = [(140 + 200 * i, 450, BOX, BOX) for i in range(5)] + \
        [(440, 1000, BOX, BOX), (640, 1000, BOX, BOX)]                     # x, y, w, h


@dataclass
class Sheet:
    """One sheet. `true_marks`/`written_total`/`is_miscounted` are HIDDEN labels: the
    learning loops only read `images`; labels are used for audits and dev/test scoring."""
    images: np.ndarray          # (7, 28, 28) uint8, MNIST-style (white ink on black)
    true_marks: np.ndarray      # (5,) the marks the teacher actually gave
    written_total: int          # the total as written (wrong on miscounted sheets)
    is_miscounted: bool         # hidden: written_total != sum(true_marks)
    split: str
    id: str
    mnist_idx: np.ndarray       # (7,) which MNIST image each box came from

    @property
    def true_digits(self) -> np.ndarray:
        """The 7 digits really written: 5 marks + tens and ones of the written total."""
        return np.concatenate([self.true_marks, [self.written_total // 10, self.written_total % 10]])


@dataclass
class SheetSet:
    """A whole split stored as arrays (fast); `sheet(i)` gives a single Sheet view."""
    split: str
    images: np.ndarray          # (N,7,28,28) uint8
    true_marks: np.ndarray      # (N,5)
    written_total: np.ndarray   # (N,)
    is_miscounted: np.ndarray   # (N,) bool
    mnist_idx: np.ndarray       # (N,7)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def digits(self) -> np.ndarray:
        """(N,7) hidden true digits (never read by the training loops)."""
        t = self.written_total
        return np.concatenate([self.true_marks, (t // 10)[:, None], (t % 10)[:, None]], axis=1)

    def sheet(self, i: int) -> Sheet:
        """Return sheet i as a Sheet dataclass."""
        return Sheet(self.images[i], self.true_marks[i], int(self.written_total[i]),
                     bool(self.is_miscounted[i]), self.split, f"{self.split}-{i:05d}", self.mnist_idx[i])

    def subset(self, idx) -> "SheetSet":
        """Select sheets by index array."""
        return SheetSet(self.split, self.images[idx], self.true_marks[idx], self.written_total[idx],
                        self.is_miscounted[idx], self.mnist_idx[idx])


class DigitPool:
    """Hands out each MNIST image at most once, so no image lands in two splits."""

    def __init__(self, labels: np.ndarray, rng: np.random.Generator):
        self.by_class = {c: [int(i) for i in rng.permutation(np.where(labels == c)[0])] for c in range(10)}

    def take(self, digit: int) -> int:
        """Pop one unused image index of class `digit`."""
        if not self.by_class[digit]:
            raise RuntimeError(f"ran out of MNIST images of class {digit}; shrink split sizes")
        return self.by_class[digit].pop()


def _wrong_total(true_sum: int, rng: np.random.Generator) -> int:
    """A plausible human error: off by 1-3, always within 0..35 and != the true sum."""
    while True:
        t = true_sum + int(rng.choice([-3, -2, -1, 1, 2, 3]))
        if 0 <= t <= sum(MAXIMA):
            return t


def generate_sheets(n: int, split: str, images: np.ndarray, pool: DigitPool,
                    rng: np.random.Generator, miscount_rate: float) -> SheetSet:
    """Make n synthetic sheets: valid marks, correct total, ~miscount_rate wrong totals."""
    marks = np.stack([rng.integers(0, m + 1, size=n) for m in MAXIMA], axis=1)
    totals = marks.sum(axis=1)
    miscounted = rng.random(n) < miscount_rate          # inject genuine human-style errors
    for i in np.where(miscounted)[0]:
        totals[i] = _wrong_total(int(totals[i]), rng)
    digits = np.concatenate([marks, (totals // 10)[:, None], (totals % 10)[:, None]], axis=1)
    idx = np.array([[pool.take(int(d)) for d in row] for row in digits])
    return SheetSet(split, images[idx], marks, totals, miscounted, idx)


def _font(size: int) -> ImageFont.ImageFont:
    """Default PIL font at a given size."""
    return ImageFont.load_default(size=size)


def blank_template() -> Image.Image:
    """The printable empty mark sheet (4 corner markers, Q1..Q5 boxes, TOTAL boxes)."""
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    for cx, cy in CORNER_CENTERS:
        d.rectangle([cx - MARKER // 2, cy - MARKER // 2, cx + MARKER // 2, cy + MARKER // 2], fill=0)
    d.text((W // 2, 260), "EvoMarks mark sheet", font=_font(56), fill=0, anchor="mm")
    d.text((W // 2, 330), "One digit per box, write clearly in black ink", font=_font(30), fill=90, anchor="mm")
    for i, (x, y, w, h) in enumerate(BOXES):
        d.rectangle([x, y, x + w, y + h], outline=0, width=4)
        if i < N_QUESTIONS:
            d.text((x + w // 2, y - 40), f"Q{i + 1}", font=_font(40), fill=0, anchor="mm")
            d.text((x + w // 2, y + h + 30), f"max {MAXIMA[i]}", font=_font(30), fill=60, anchor="mm")
    d.text((W // 2, 930), "TOTAL (2 digits, e.g. 07)", font=_font(40), fill=0, anchor="mm")
    return img


def render_sheet(sheet: Sheet | SheetSet, i: int = 0) -> Image.Image:
    """Draw a sheet: digit images pasted (black ink on white) into the template boxes."""
    images = sheet.images if isinstance(sheet, Sheet) else sheet.images[i]
    img = blank_template()
    for (x, y, w, h), digit in zip(BOXES, images):
        ink = Image.fromarray(255 - digit).resize((w - 32, h - 32), Image.BILINEAR)
        img.paste(ink, (x + 16, y + 16))
    return img
