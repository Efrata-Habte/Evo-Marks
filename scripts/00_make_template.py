"""Step 0: export the printable mark-sheet template (PNG + A4 PDF) with corner markers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ROOT
from evomarks.sheets import blank_template


def main() -> None:
    """Write template/evomarks_template.png and .pdf (150 dpi => exactly A4)."""
    out = ROOT / "template"
    out.mkdir(exist_ok=True)
    img = blank_template()
    img.save(out / "evomarks_template.png")
    img.convert("RGB").save(out / "evomarks_template.pdf", resolution=150.0)
    print("saved", out / "evomarks_template.png", "and .pdf - print on A4, fill by hand, photograph flat")


if __name__ == "__main__":
    main()
