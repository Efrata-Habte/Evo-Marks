"""Step 8: build docs/EvoMarks_slides.pptx (13 slides, with speaker notes) from results/figures and the report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from config import ROOT

FIG = ROOT / "results" / "figures"
INK, MUTED, ACCENT = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x64, 0x74, 0x8B), RGBColor(0x25, 0x63, 0xEB)
NAVY, BG, CARD, LINE = RGBColor(0x0F, 0x1B, 0x3D), RGBColor(0xF4, 0xF6, 0xFB), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xE2, 0xE8, 0xF0)
SKY, WHITE, CODE_BG = RGBColor(0x93, 0xC5, 0xFD), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x11, 0x18, 0x27)
FONT = "Calibri"
W, H = 13.333, 7.5

CODE = ("# parse -> change one node -> unparse\n\n"
        "tree = ast.parse(source)\n"
        "# e.g. STEP_SCALE = 0.3 -> 0.45\n"
        "node.value = ast.Constant(0.45)\n"
        "new_source = ast.unparse(tree)\n\n"
        "# illustrative; see evomarks/code_editor.py")

SLIDES = [
    dict(kicker="Title", title="EvoMarks: a system that checks handwritten mark sheets and improves itself safely",
         bullets=["Reads the marks and the total, checks that they add up, and flags the sheet if not. Offline and small.",
                  "Track A: a bounded self-improvement loop with an independent judge. Track B: shrinking the model by distillation."],
         figs=["template/evomarks_template.png"],
         notes="Say: people type exam marks by hand and totals are often wrong. EvoMarks reads the digits, checks the arithmetic, "
               "and flags sheets that do not add up. The interesting part is not the digit reader, it is how the system improves "
               "itself without cheating. Two tracks: A is the self-improvement loop, B is making the model small."),
    dict(kicker="The core idea", title="Arithmetic is the free, independent judge",
         bullets=["A sheet has 5 marks and a total. If the marks fit their maximum and add up to the total, the reading is very likely right.",
                  "No human labels are needed to know that. The rule itself is the supervision.",
                  "So the system can look at thousands of unlabeled sheets and keep only the ones the rule trusts.",
                  "The rule never sees the model's confidence or the true labels, so the model cannot talk its way past it."],
         figs=[],
         notes="This is the one idea to land. Labels are expensive, but arithmetic is free. A reading that satisfies the sum check "
               "is almost certainly correct, so we can turn unlabeled sheets into trusted training data. Stress that the verifier is "
               "independent: it uses a different kind of evidence from the model (a rule, not a probability). That independence is what "
               "stops the model from reinforcing its own mistakes."),
    dict(kicker="Design decision 1", title="The judge is fixed; the thing being improved is not",
         bullets=["Two roles are kept apart: the part that proposes changes, and the part that decides whether a change is kept.",
                  "The evaluator, the verifier, the data and the tests are protected. Candidates are not allowed to mention them.",
                  "Test sheets come from unseen writers and are read once, only for reporting. Nothing is tuned on them.",
                  "Every version is archived, accepted or rejected, with its reason. A bad change is rolled back, never lost."],
         figs=[],
         notes="If the system could change its own judge, it would just learn to pass. So the judge is frozen, its files are hashed "
               "before and after every run, and the test set is touched once. Because everything is archived with a reason, you can "
               "audit why any version was kept or thrown away."),
    dict(kicker="Design decision 2", title="Edit the code as a tree, not as text",
         bullets=["The improver reads its own code as a tree, changes one value or list, and writes it back as new Python.",
                  "That is how the system rewrites its own source safely without an LLM.",
                  "Every candidate is valid Python by construction: no syntax errors, no half-written files.",
                  "Each edit is one small, named change (tweak a constant, add or remove a block, resize the model), so a result can be traced to its cause."],
         figs=[], code=CODE,
         notes="Use the sentence on the slide almost word for word. Explain the contrast: an LLM writing code can produce anything, "
               "including broken or dangerous code. Here the improver parses the file into a tree with Python's ast module, changes "
               "exactly one value or list, and turns the tree back into source. So it is safe, cheap, reproducible, and every change "
               "is explainable. The trade-off, which we say honestly later, is that the space of edits is small."),
    dict(kicker="Design decision 3", title="Small edit space, many guard-rails",
         bullets=["Edits are limited to a short list: constants, whitelisted preprocessing and augmentation blocks, and model size. Each constant has hard bounds.",
                  "Before running, a static check allows only a whitelist of imports and blocks risky names such as file, system and eval access.",
                  "Each candidate runs in a separate process with a time limit, under one fixed small budget.",
                  "Honest limit: this is a whitelist plus process isolation, not an operating-system sandbox."],
         figs=[],
         notes="Safety was designed in, not added at the end. Three layers: the editor can only produce a limited kind of change, "
               "a static check rejects forbidden code, and the run happens in a time-limited subprocess. Be upfront that this is not "
               "a true OS sandbox. Being clear about that limit is part of the design."),
    dict(kicker="Design decision 4", title="Three levels, each changes something different",
         bullets=["Level 0 changes the model's weights: retrain on sheets the sum check trusts, keep only if a held-out set improves.",
                  "Level 1 changes the pipeline's code: the improver proposes an edit, the fixed evaluator accepts or rejects it.",
                  "Level 2 changes the improver itself: it edits its own settings and is kept only if its edits help the pipeline more than before.",
                  "Each level is judged by something it cannot edit, using data the level below never saw."],
         figs=[],
         notes="Walk up the ladder. Level 0 is ordinary self-training, but gated. Level 1 is where code gets rewritten. Level 2 is the "
               "recursive step: the thing that makes the edits is itself edited, and judged on separate data (meta_dev) so it cannot "
               "overfit to the set used by Level 1. The idea follows the STOP paper: an improver is only better if it produces better improvements."),
    dict(kicker="Results: Level 0", title="A rule beats the model's own confidence",
         bullets=["Choosing training sheets by the sum check gives cleaner pseudo-labels than choosing by the model's confidence.",
                  "Confidence-based selection feeds the model its own mistakes (confirmation bias). The rule does not.",
                  "Accuracy rose from 0.887 to 0.944 on unseen test writers, using no extra labels."],
         figs=["l0_verifier_vs_confidence.png", "l0_accuracy_by_round.png"],
         notes="This is the clear success. Compare two ways of picking pseudo-labels: trust confident predictions, or trust sheets that "
               "add up. The rule is cleaner (0.994 vs 0.968 precision), and the final model is better. The reason is "
               "independence: confidence comes from the same model that might be wrong."),
    dict(kicker="Results: Level 1", title="Mostly rejection; one small gain, honestly measured",
         bullets=["Of 20 proposed code edits, 2 were accepted and 18 were rejected. Most random edits make things worse.",
                  "We measured the noise of the gate itself: retraining the same version varies more than the acceptance margin.",
                  "So we compared versions seed by seed. The gain is small, probably real, and disappears at full training budget."],
         figs=["l1_evolution_timeline.png", "l1_edit_effectiveness.png"],
         notes="Do not oversell. The loop works as designed: it is picky. But the check on noise matters: a single evaluation can pass "
               "or fail a candidate by chance. Paired by seed the evolved version is +1.05 points on dev at the small budget, but at the "
               "full budget the test difference (0.9455 vs 0.9436) is within noise. The loop optimised the cheap evaluator."),
    dict(kicker="Results: Level 2", title="A null result, and we know why",
         bullets=["The improver proposed 3 edits to itself and none were accepted, so Level 1+2 equals Level 1 alone.",
                  "The gate measures usefulness from one fixed parent, so some edits (such as parent selection) cannot change the score.",
                  "This is a limit of our test, not proof that recursion cannot work."],
         figs=["l2_improver_utility.png", "l2_ablation.png"],
         notes="Say this plainly: Level 2 did not help. The valuable part is the diagnosis. Our utility measure fixes the parent, so "
               "an edit to how parents are chosen is invisible to it. The comparison between Level 1+2 and Level 1 is therefore "
               "comparing identical processes. A fairer gate would be the next step."),
    dict(kicker="Track B: distillation", title="Shrink the model; the average hides the tail",
         bullets=["A small student learns from the big teacher's soft probabilities, on the same data as a normal student, so only the targets differ.",
                  "Result: about 3x smaller and 2.5 to 4 times faster, for roughly 3.6 points less accuracy.",
                  "Distilling helped little over plain training, and the evolved teacher was no better a teacher than the original.",
                  "Averages hide failures: the student is much worse on digit 9."],
         figs=["kd_accuracy_vs_latency.png", "kd_per_digit_gap.png"],
         notes="Link the tracks: the loop produces verified pseudo-labels, and those become the student's training data. Noise in those "
               "labels flows into the student. Keep the comparison fair: same data, only the targets differ. Be honest that distillation "
               "gave a tiny gain, and that an improved teacher did not give a better student. Mention digit 9 as an example of why "
               "per-class results matter."),
    dict(kicker="Demo", title="Reading a sheet",
         bullets=["Run: streamlit run app/streamlit_app.py, then the Try it page. Upload a photo or use a random synthetic sheet.",
                  "Show: digits read, sum check, flag or no flag.",
                  "Then show a version next to its parent: code diff, metric change, per-digit bars. This is the audit trail.",
                  "Backup if the app fails: python demo.py photo.jpg"],
         figs=[],
         notes="Keep the demo short. The point to show is the audit trail: you can see exactly what code changed between two versions and "
               "what it did to accuracy. A flag means a human should look at the sheet, not that the sheet is wrong."),
    dict(kicker="Honest limits", title="What did not work and what this does not show",
         bullets=["Most edits hurt, and the early gain came from a cheap evaluator. Lesson: measure the noise of your gate.",
                  "Wrong digits that still add up pass the check. Flags are mostly misreads, so a flag means please check.",
                  "Sheets are built from MNIST digits, cleaner than phone photos. The real-photo test was not run.",
                  "This is bounded, sandboxed self-improvement, not open-ended."],
         figs=["system_sheet_metrics.png"],
         notes="Own the weaknesses before anyone asks. The real-handwriting test was not run, so absolute accuracies are optimistic. The "
               "edit space is small, there are few seeds, and the sandbox is not OS-level. What the project does show is a careful design: "
               "independent judge, constrained edits, full audit trail, and honest measurement of noise."),
    dict(kicker="Takeaways", title="What to remember",
         bullets=["Use an independent judge that the system cannot edit or fool.",
                  "Rewrite code as a tree so every change is valid, small and traceable.",
                  "Keep a full archive and a rollback, and report null results and noise honestly.",
                  "Next: test on real photographs and fix the Level 2 gate."],
         figs=[],
         notes="Close on the four design principles, not the numbers. Then name the two concrete next steps and invite questions."),
]


def box(slide, left, top, width, height, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    """Filled shape without shadow."""
    sh = slide.shapes.add_shape(shape, Inches(left), Inches(top), Inches(width), Inches(height))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb, sh.line.width = line, Pt(1)
    if radius is not None:
        sh.adjustments[0] = radius
    sh.shadow.inherit = False
    return sh


def add_text(slide, left, top, width, height, text, size, bold=False, color=INK, anchor=MSO_ANCHOR.TOP,
             align=PP_ALIGN.LEFT, font=FONT):
    """One text box with a single paragraph."""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text, p.alignment = text, align
    p.font.size, p.font.bold, p.font.color.rgb, p.font.name = Pt(size), bold, color, font
    return tf


def add_card(slide, left, top, width, height, number, text, size):
    """White rounded card with a numbered badge and the text."""
    box(slide, left, top, width, height, CARD, LINE, MSO_SHAPE.ROUNDED_RECTANGLE, 0.08)
    box(slide, left, top + 0.18, 0.09, height - 0.36, ACCENT)
    badge = box(slide, left + 0.28, top + (height - 0.46) / 2, 0.46, 0.46, ACCENT, shape=MSO_SHAPE.OVAL)
    tf = badge.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text, p.alignment = str(number), PP_ALIGN.CENTER
    p.font.size, p.font.bold, p.font.color.rgb, p.font.name = Pt(16), True, WHITE, FONT
    add_text(slide, left + 0.95, top + 0.12, width - 1.2, height - 0.24, text, size, anchor=MSO_ANCHOR.MIDDLE)


def add_picture_panel(slide, path: Path, left, top, max_w, max_h) -> None:
    """Picture on a white panel, scaled to fit, aspect ratio kept and centered."""
    box(slide, left, top, max_w, max_h, CARD, LINE, MSO_SHAPE.ROUNDED_RECTANGLE, 0.03)
    w, h = Image.open(path).size
    scale = min((max_w - 0.2) / w, (max_h - 0.2) / h)
    pw, ph = w * scale, h * scale
    slide.shapes.add_picture(str(path), Inches(left + (max_w - pw) / 2), Inches(top + (max_h - ph) / 2),
                             Inches(pw), Inches(ph))


def add_code_panel(slide, left, top, width, height, code: str) -> None:
    """Dark panel with monospace code; comment lines in muted blue."""
    box(slide, left, top, width, height, CODE_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
    tf = slide.shapes.add_textbox(Inches(left + 0.35), Inches(top + 0.3), Inches(width - 0.7), Inches(height - 0.6)).text_frame
    tf.word_wrap = True
    for i, line in enumerate(code.split(chr(10))):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name, p.font.size = "Consolas", Pt(16)
        p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8) if line.startswith("#") else RGBColor(0xE2, 0xE8, 0xF0)


def frame(slide, d, n, total) -> None:
    """Background, navy header band with kicker and title, footer."""
    box(slide, 0, 0, W, H, BG)
    box(slide, 0, 0, W, 1.55, NAVY)
    box(slide, 0, 1.55, W, 0.06, ACCENT)
    add_text(slide, 0.6, 0.28, 12, 0.3, d["kicker"].upper(), 13, True, SKY)
    add_text(slide, 0.6, 0.62, 12.1, 0.8, d["title"], 28, True, WHITE, MSO_ANCHOR.TOP)
    add_text(slide, 0.6, 7.08, 6, 0.25, "EvoMarks", 11, True, MUTED)
    add_text(slide, W - 1.6, 7.08, 1.0, 0.25, f"{n} / {total}", 11, False, MUTED, align=PP_ALIGN.RIGHT)


def title_slide(slide, d, paths) -> None:
    """Dark full-bleed opening slide."""
    box(slide, 0, 0, W, H, NAVY)
    box(slide, 0.6, 1.3, 0.9, 0.08, ACCENT)
    add_text(slide, 0.6, 1.55, 6.4, 0.4, "TRACK A  +  TRACK B", 14, True, SKY)
    add_text(slide, 0.6, 2.0, 6.4, 2.5, d["title"], 34, True, WHITE)
    y = 4.75
    for b in d["bullets"]:
        add_text(slide, 0.6, y, 6.4, 1.0, b, 16, False, RGBColor(0xCB, 0xD5, 0xE1))
        y += 1.0
    if paths:
        box(slide, 7.7, 0.9, 5.0, 5.7, WHITE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.03)
        w, h = Image.open(paths[0]).size
        scale = min(4.6 / w, 5.3 / h)
        slide.shapes.add_picture(str(paths[0]), Inches(7.7 + (5.0 - w * scale) / 2), Inches(0.9 + (5.7 - h * scale) / 2),
                                 Inches(w * scale), Inches(h * scale))


def main() -> None:
    """Write the deck."""
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    total = len(SLIDES)
    for n, d in enumerate(SLIDES, 1):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        paths = [FIG / f if not f.startswith("template") else ROOT / f for f in d["figs"]]
        bullets = d["bullets"]
        if n == 1:
            title_slide(s, d, paths)
        else:
            frame(s, d, n, total)
            top, bottom = 1.95, 6.95
            if paths or "code" in d:  # stacked cards left, visual right
                gap = 0.2
                ch = (bottom - top - gap * (len(bullets) - 1)) / len(bullets)
                size = 16 if len(bullets) >= 4 else 17
                for i, b in enumerate(bullets):
                    add_card(s, 0.6, top + i * (ch + gap), 6.0, ch, i + 1, b, size)
                if "code" in d:
                    add_code_panel(s, 6.9, top, 5.85, bottom - top, d["code"])
                elif len(paths) == 1:
                    add_picture_panel(s, paths[0], 6.9, top, 5.85, bottom - top)
                else:
                    ph = (bottom - top - 0.2) / 2
                    add_picture_panel(s, paths[0], 6.9, top, 5.85, ph)
                    add_picture_panel(s, paths[1], 6.9, top + ph + 0.2, 5.85, ph)
            else:  # 2 x 2 grid of cards
                gap = 0.25
                cw, ch = (12.13 - gap) / 2, (bottom - top - gap) / 2
                for i, b in enumerate(bullets):
                    add_card(s, 0.6 + (i % 2) * (cw + gap), top + (i // 2) * (ch + gap), cw, ch, i + 1, b, 19)
        s.notes_slide.notes_text_frame.text = d["notes"]
    for suffix in ["", "_new", "_v3", "_v4", "_v5"]:  # earlier decks may be open in PowerPoint
        out = ROOT / "docs" / f"EvoMarks_slides{suffix}.pptx"
        try:
            prs.save(out)
            break
        except PermissionError:
            continue
    else:
        raise SystemExit("close the open deck in PowerPoint and rerun")
    print(f"wrote {out} ({total} slides)")


if __name__ == "__main__":
    main()
