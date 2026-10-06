"""Synthetic exam dataset generator with ground truth.

Produces, per student:
  * a text document (printed answer sheet)
  * an image "handwriting"-style answer sheet (Hindi + English) that defeats
    naive OCR on purpose, to exercise the repair/escalation path
  * a text-based PDF
  * a scanned PDF (image-only)
  * a mixed PDF (page 1 text layer, page 2 scan)

and a `ground_truth.json` with the expected student id, per-question answer
text and page-level text, so reader/extraction accuracy can be measured.

Usage:
    python -m scripts.gen_synthetic --out /data/demo --students 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.schemas import SourceType

EXAM = {
    "name": "Half-Yearly Science Test",
    "subject": "Science",
    "exam_date": "2026-09-20",
    "language": "mixed",
    "questions": [
        {
            "q_no": "Q1",
            "text": "What is 7 x 8?",
            "text_hindi": "7 x 8 कितना होता है?",
            "max_score": 1,
            "strategy": "numeric",
            "answer_key": "56",
        },
        {
            "q_no": "Q2",
            "text": "State Newton's first law of motion in one sentence.",
            "text_hindi": "न्यूटन का पहला नियम एक वाक्य में लिखें।",
            "max_score": 3,
            "strategy": "rubric_llm",
            "answer_key": (
                "An object remains at rest or in uniform motion unless acted upon by an "
                "external force."
            ),
        },
        {
            "q_no": "Q3",
            "text": "Give two differences between mass and weight.",
            "text_hindi": "द्रव्यमान और भार में दो अंतर लिखें।",
            "max_score": 3,
            "strategy": "rubric_llm",
            "answer_key": (
                "Mass is the amount of matter and is constant; weight is the gravitational "
                "force on that mass and changes with gravity. Mass unit: kg, weight unit: newton."
            ),
        },
        {
            "q_no": "Q4",
            "text": "Why do we see the Moon?",
            "text_hindi": "हमें चंद्रमा क्यों दिखाई देता है?",
            "max_score": 2,
            "strategy": "rubric_llm",
            "answer_key": "The Moon reflects sunlight towards the Earth, so it appears lit.",
        },
    ],
}

RUBRIC = {
    "name": "Science 3-criteria rubric",
    "description": "Correctness first; completeness and reasoning add marks.",
    "criteria": [
        {
            "name": "correctness",
            "weight": 0.5,
            "description": "Scientific correctness of the answer.",
            "knockout": True,
            "knockout_below_fraction": 0.0,
        },
        {
            "name": "completeness",
            "weight": 0.3,
            "description": "All expected points are present.",
        },
        {
            "name": "reasoning",
            "weight": 0.2,
            "description": "Clear explanation or working shown.",
        },
    ],
}

STUDENTS = [
    {"external_id": "STU001", "name": "Aarav Sharma", "batch": "Batch A"},
    {"external_id": "STU002", "name": "Priya Verma", "batch": "Batch A"},
    {"external_id": "STU003", "name": "Rahul Mehta", "batch": "Batch A"},
    {"external_id": "STU004", "name": "Sneha Iyer", "batch": "Batch B"},
    {"external_id": "STU005", "name": "Imran Khan", "batch": "Batch B"},
]

# Per-student answer variants, from perfect to poor, Hindi and English mixed.
ANSWER_SETS = {
    "STU001": {
        "Q1": "56",
        "Q2": "An object stays at rest or keeps moving uniformly unless an external force acts on it.",
        "Q3": "Mass is the quantity of matter in a body and stays the same; weight is the force of gravity on the body and changes from place to place.",
        "Q4": "We see the Moon because it reflects sunlight that reaches the Earth.",
    },
    "STU002": {
        "Q1": "56",
        "Q2": "Objects remain at rest until a force is applied.",
        "Q3": "Mass is matter and weight is gravity force.",
        "Q4": "Because the Moon is bright.",
    },
    "STU003": {
        "Q1": "54",
        "Q2": "न्यूटन का पहला नियम: वस्तु अपनी अवस्था में तब तक रहती है जब तक बाहरी बल न लगे।",
        "Q3": "द्रव्यमान पदार्थ की मात्रा है, भार गुरुत्वाकर्षण बल है। द्रव्यमान किलोग्राम में और भार न्यूटन में मापा जाता है।",
        "Q4": "चंद्रमा सूर्य की रोशनी को परावर्तित करता है, इसलिए हमें दिखाई देता है।",
    },
    "STU004": {
        "Q1": "56",
        "Q2": "Force makes things move.",
        "Q3": "Weight is more than mass.",
        "Q4": "I do not know.",
    },
    "STU005": {
        "Q1": "56",
        "Q2": "Newton first law says object in motion stays in motion.",
        "Q3": "द्रव्यमान और भार अलग हैं।",
        "Q4": "चंद्रमा रोशनी देता है।",
    },
}

HINDI_LABELS = {
    "name": "नाम",
    "roll": "अनुक्रमांक",
    "exam": "परीक्षा",
}


DEVANAGARI_RANGE = (0x0900, 0x097F)
DEVANAGARI_FONT = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def find_font(size: int = 30):
    """Return (latin_font, devanagari_font_or_None) for the given size."""
    latin = ImageFont.truetype(LATIN_FONT, size) if Path(LATIN_FONT).exists() else ImageFont.load_default()
    deva = (
        ImageFont.truetype(DEVANAGARI_FONT, size)
        if Path(DEVANAGARI_FONT).exists()
        else None
    )
    return latin, deva


def _font_for(char: str, latin, deva):
    if deva is not None and DEVANAGARI_RANGE[0] <= ord(char) <= DEVANAGARI_RANGE[1]:
        return deva
    return latin


def draw_mixed_text(draw: ImageDraw.ImageDraw, position, text: str, latin, deva, *, fill=(10, 10, 10)):
    """Draw text mixing Latin and Devanagari, switching fonts per script run."""
    x, y = position
    if deva is None:
        draw.text((x, y), text, font=latin, fill=fill)
        return
    run = ""
    run_font = latin
    for char in text:
        font = _font_for(char, latin, deva)
        if font is not run_font and run:
            draw.text((x, y), run, font=run_font, fill=fill)
            x += draw.textlength(run, font=run_font)
            run = ""
        run_font = font
        run += char
    if run:
        draw.text((x, y), run, font=run_font, fill=fill)


def render_answer_sheet(
    student: dict,
    answers: dict[str, str],
    *,
    handwritten: bool,
    path: Path,
    width: int = 1240,
    height: int = 1754,
) -> None:
    """Render an answer sheet. Handwritten mode adds jitter/slant/noise."""
    background = (252, 250, 242) if handwritten else (255, 255, 255)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    font, deva = find_font(30)
    small, small_deva = find_font(24)
    header_font, header_deva = find_font(34)

    if handwritten:
        # ruled paper effect
        for y in range(150, height, 56):
            draw.line([(40, y), (width - 40, y)], fill=(205, 215, 230), width=1)
        margin_x = 46
    else:
        margin_x = 60

    draw_mixed_text(
        draw, (margin_x, 50), f"{EXAM['name']} — {EXAM['subject']}", header_font, header_deva,
        fill=(20, 20, 60),
    )
    draw_mixed_text(
        draw,
        (margin_x, 100),
        f"{HINDI_LABELS['name']}: {student['name']}   {HINDI_LABELS['roll']}: {student['external_id']}",
        small,
        small_deva,
        fill=(40, 40, 40),
    )
    draw.line([(margin_x, 140), (width - margin_x, 140)], fill=(120, 120, 120), width=2)

    y = 170
    for question in EXAM["questions"]:
        q_no = question["q_no"]
        label = f"{q_no}. {question['text']}"
        if handwritten:
            _draw_jittered_mixed(draw, (margin_x, y), label, font, deva, seed=q_no)
        else:
            draw_mixed_text(draw, (margin_x, y), label, font, deva)
        y += 48
        answer = answers.get(q_no) or "(no answer)"
        wrapped = _wrap(answer, 52)
        for line in wrapped:
            if handwritten:
                _draw_jittered_mixed(draw, (margin_x + 20, y), line, font, deva, seed=line)
            else:
                draw_mixed_text(draw, (margin_x + 20, y), line, font, deva)
            y += 44
        y += 34

    if handwritten:
        image = image.rotate(-0.7, resample=Image.BICUBIC, fillcolor=background, expand=False)
        image = image.filter(ImageFilter.GaussianBlur(radius=0.4))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_jittered_mixed(
    draw: ImageDraw.ImageDraw, position, text: str, latin, deva, *, seed: str, fill=None
) -> None:
    rng = random.Random(seed)
    x, y = position
    for char in text:
        font = _font_for(char, latin, deva)
        ink = fill or (
            15 + rng.randint(0, 40),
            15 + rng.randint(0, 30),
            60 + rng.randint(0, 50),
        )
        draw.text(
            (x + rng.uniform(-1.5, 1.5), y + rng.uniform(-3.0, 3.0)),
            char,
            font=font,
            fill=ink,
        )
        x += draw.textlength(char, font=font) + rng.uniform(-0.6, 0.6)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def make_text_doc(student: dict, answers: dict[str, str], path: Path) -> None:
    lines = [f"{EXAM['name']} — {EXAM['subject']}", f"Name: {student['name']}", f"Roll: {student['external_id']}", ""]
    for question in EXAM["questions"]:
        q_no = question["q_no"]
        lines.append(f"{q_no}. {question['text']}")
        lines.append(f"Ans: {answers.get(q_no, '(no answer)')}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_pdf(student: dict, answers: dict[str, str], path: Path, *, mode: str) -> None:
    """mode: text | scanned | mixed"""
    import pymupdf

    doc = pymupdf.open()
    if mode in ("text", "mixed"):
        page = doc.new_page()
        text = [f"{EXAM['name']} — {student['name']} ({student['external_id']})", ""]
        for question in EXAM["questions"]:
            q_no = question["q_no"]
            text.append(f"{q_no}. {question['text']}")
            text.append(f"Ans: {answers.get(q_no, '(no answer)')}")
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), "\n".join(text), fontsize=12)
        if mode == "mixed":
            image = _render_page_image(student, answers)
            page2 = doc.new_page()
            page2.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=image)
    else:  # scanned: single image-only page
        image = _render_page_image(student, answers)
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=image)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def _render_page_image(student: dict, answers: dict[str, str]) -> bytes:
    import io

    from app.pipeline.preprocessing import to_jpeg_bytes

    image = Image.new("RGB", (1240, 1754), (250, 248, 240))
    draw = ImageDraw.Draw(image)
    font, deva = find_font(28)
    small, small_deva = find_font(22)
    draw_mixed_text(
        draw, (60, 56), f"Scanned answer sheet — {student['name']}", font, deva, fill=(20, 20, 20)
    )
    y = 120
    for question in EXAM["questions"]:
        q_no = question["q_no"]
        _draw_jittered_mixed(
            draw, (60, y), f"{q_no}. {question['text']}", font, deva,
            seed=q_no + student["external_id"],
        )
        y += 50
        for line in _wrap(answers.get(q_no, "(no answer)"), 46):
            _draw_jittered_mixed(draw, (80, y), line, font, deva, seed=line)
            y += 44
        y += 30
    return to_jpeg_bytes(image.convert("RGB"), max_side=1600, quality=80)


def generate(out: Path, students: list[dict] | None = None) -> dict:
    students = students or STUDENTS
    out.mkdir(parents=True, exist_ok=True)
    truth: dict = {"exam": EXAM, "rubric": RUBRIC, "students": {}}

    for index, student in enumerate(students):
        answers = ANSWER_SETS.get(
            student["external_id"],
            {q["q_no"]: f"Answer for {q['q_no']}" for q in EXAM["questions"]},
        )
        base = out / "Class 8" / student["batch"] / student["external_id"]
        handwritten = index % 2 == 0
        render_answer_sheet(student, answers, handwritten=handwritten, path=base / "answer_sheet.jpg")
        make_text_doc(student, answers, base / "answer_sheet.txt")
        make_pdf(student, answers, base / "answer_sheet_text.pdf", mode="text")
        make_pdf(student, answers, base / "answer_sheet_scanned.pdf", mode="scanned")
        make_pdf(student, answers, base / "answer_sheet_mixed.pdf", mode="mixed")

        truth["students"][student["external_id"]] = {
            "name": student["name"],
            "batch": student["batch"],
            "answers": answers,
            "files": [
                "answer_sheet.jpg",
                "answer_sheet.txt",
                "answer_sheet_text.pdf",
                "answer_sheet_scanned.pdf",
                "answer_sheet_mixed.pdf",
            ],
        }

    # duplicate file + blank page: exercise the idempotency and empty-validation paths
    dup_source = out / "Class 8" / "Batch A" / "STU001" / "answer_sheet.jpg"
    dup_target = out / "Class 8" / "Batch A" / "STU001" / "answer_sheet_copy.jpg"
    dup_target.write_bytes(dup_source.read_bytes())

    blank = Image.new("RGB", (1240, 1754), (255, 255, 255))
    blank_dir = out / "Class 8" / "Batch B" / "STU005"
    blank.save(blank_dir / "blank_page.jpg")

    (out / "ground_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic exam data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--students", type=int, default=5)
    args = parser.parse_args()
    students = STUDENTS[: args.students]
    truth = generate(args.out, students)
    print(
        f"Generated {len(truth['students'])} students x 5 documents + duplicate + blank "
        f"under {args.out}"
    )


if __name__ == "__main__":
    main()
