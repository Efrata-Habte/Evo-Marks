"""CLI demo: read a photographed mark sheet with any version.
   python demo.py photo.jpg [--version best|v000|student|vNNN]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MAXIMA, ROOT
from evomarks.archive import default_readers, load_reader
from evomarks.photo import read_photo
from evomarks.verifier import verify


def main() -> None:
    """Read one photo and print the digits, the sum check and the verdict."""
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--version", default="best")
    args = ap.parse_args()
    readers = default_readers(ROOT / "archive", ROOT / "results")
    if args.version not in readers:       # any archived pipeline version, e.g. v003 (evolution-budget weights)
        readers[args.version] = load_reader(ROOT / "archive" / args.version, args.version)
    res = read_photo(args.photo)
    for w in res.warnings:
        print("WARNING:", w)
    probs = readers[args.version].probs(res.crops)
    digits = probs.argmax(1)
    marks, total = [int(d) for d in digits[:5]], int(digits[5]) * 10 + int(digits[6])
    for i, (d, p) in enumerate(zip(digits, probs.max(1))):
        print(f"box {i + 1}: {d}  (confidence {p:.2f})")
    v = verify(marks, total, MAXIMA)
    print(f"marks {marks} sum {sum(marks)} | written total {total} | "
          f"{'OK: total checks out' if v.accepted else 'CHECK THIS SHEET (' + v.reason + ')'}")


if __name__ == "__main__":
    main()
