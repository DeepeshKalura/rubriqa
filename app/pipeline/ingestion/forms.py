"""Form parsers for both Google Forms variants used by the program.

Two known shapes are supported (auto-detected by header lookup):

  1. "Himachal" style form — students type their name, school, etc. directly.
     Expected headers:
       Timestamp, Email Address, Full Name, Mobile/WhatsApp, School Name,
       Class, State, District, School (selected), Language, Theme,
       AI description, [Wi-fi/Broadband/Computer/Tablet/Laptop/Mobile],
       Situation, Proposed AI Solution, Declaration of Originality,
       Drive link (column 26 in the client's sheet).

  2. "Ashramshala" (Marathi) style form — name selected from a list, division,
     project office, Ashramshala chosen. Expected headers:
       Timestamp, Name (selected), Division, Project Office, Ashramshala,
       Class, [Wi-fi/Broadband/Computer/Smartboard/TV/Projector],
       Drive link, Submission count, PO Name, School Name.

Each parser returns a normalised `ParsedFormResponse` ready to become an Entry.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ------- resource keys (boolean per entry) -------------------------------
HIMACHAL_RESOURCES = ("Wi-fi", "Broadband Network", "Computer", "Tablet", "Laptop", "Mobile")
ASHRAMSHALA_RESOURCES = ("Wi-fi", "Broadband Network", "Computer", "Smartboard", "TV", "Projector")

YES_TRUE = {"yes", "y", "हाँ", "हां", "हो", "true", "1", "✓", "confirm"}
NO_FALSE = {"no", "n", "नहीं", "नाही", "false", "0", "x"}

# Theme mapping: the form response uses mixed Hindi / English / Marathi labels.
# We normalise to a short slug so callers can group + tag.
THEME_SLUGS: dict[str, str] = {
    "smart agriculture":          "smart_agriculture",
    "smart agriculture and rural innovation": "smart_agriculture",
    "स्मार्ट कृषि": "smart_agriculture",
    "स्मार्ट कृषि और ग्रामीण नवाचार में ai": "smart_agriculture",
    "ai in smart agriculture & rural innovation": "smart_agriculture",
    "ai for education":                    "ai_education",
    "ai in education & learning":          "ai_education",
    "education":                          "ai_education",
    "ai for education & learning":        "ai_education",
    "health":                             "health_wellbeing",
    "health and wellbeing":               "health_wellbeing",
    "ai in health & community well-being": "health_wellbeing",
    "स्वास्थ्य और कल्याण में ai": "health_wellbeing",
    "health & community well-being":      "health_wellbeing",
    "wellbeing":                      "health_wellbeing",
    "disaster management":            "disaster_management",
    "आपदा प्रबंधन में ai":   "disaster_management",
    "आपदा प्रबंधन":            "disaster_management",
    "ai in disaster management":      "disaster_management",
    "environment":                    "environment_sustainability",
    "environment and sustainability": "environment_sustainability",
    "ai in sustainability & climate action": "environment_sustainability",
    "पर्यावरण संरक्षण में ai":   "environment_sustainability",
    "rural innovation":               "rural_innovation",
    "community safety":               "community_safety",
}


def normalise_theme(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip().lower()
    if key in THEME_SLUGS:
        return THEME_SLUGS[key]
    # fall back: turn into slug
    slug = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return slug[:80] or None


@dataclass
class ParsedFormResponse:
    """A normalised, language-agnostic representation of one Google Form row."""

    form_kind: str                                # himachal | ashramshala
    external_ref: str                             # stable id used for idempotent registration
    timestamp: str | None = None
    email: str | None = None
    full_name: str | None = None
    gender: str | None = None
    phone: str | None = None
    school_name: str | None = None
    class_name: str | None = None
    state: str | None = None
    district: str | None = None
    block_name: str | None = None
    division: str | None = None
    project_office: str | None = None
    ashramshala: str | None = None
    language: str | None = None
    theme: str | None = None
    ai_description: str | None = None
    resources: dict[str, bool] = field(default_factory=dict)
    situation: str | None = None
    proposed_solution: str | None = None
    declaration: bool = False
    consent: bool = False
    drive_url: str | None = None
    drive_file_id: str | None = None
    po_name: str | None = None
    school_name_form: str | None = None
    submission_count: int | None = None
    raw: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _yes(v: str | None) -> bool:
    """Treat any non-empty string that isn't an explicit refusal as a yes.

    The Declaration column in particular contains the student's typed
    confirmation ("I confirm that the idea shared is my own..."), which we
    accept as long as it's not a negative.
    """
    if v is None:
        return False
    txt = v.strip().lower()
    if txt in NO_FALSE:
        return False
    if txt in YES_TRUE:
        return True
    # the typed declaration text starts with "i confirm" / "confirm"
    if "confirm" in txt:
        return True
    # any non-empty value (other than an explicit refusal above) counts
    return bool(txt)


def _drive_id(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _external_ref(row_index: int, timestamp: str | None, full_name: str | None,
                  school: str | None) -> str:
    """Stable id across re-imports; falls back to row index if everything else is blank."""
    base = "|".join(filter(None, (timestamp, full_name, school or None)))
    if base:
        return base[:120]
    return f"row-{row_index:06d}"


# ---------------------------------------------------------------------------
# himachal
# ---------------------------------------------------------------------------
HIMACHAL_HEADER_ALIASES: dict[str, list[str]] = {
    "timestamp":   ["timestamp"],
    "email":       ["email address"],
    "full_name":   ["write full name in english", "full name in english"],
    "gender":      ["gender"],
    "phone":       ["write your or your parent's/teacher's whatsapp/mobile number", "mobile number"],
    "school_name": ["school name in english"],
    "class_name":  ["which class do you study in", "class"],
    "state":       ["state"],
    "district":    ["districts of himachal pradesh", "district"],
    "language":    ["please select the language in which you would like to fill out the form", "language"],
    "theme":       ["अपने प्रोजेक्ट विचार के लिए केवल एक थीम का चयन करें", "theme"],
    "ai_description": ["ai का सबसे अच्छा वर्णन क्या है", "ai description"],
    "situation":   ["situation", "परिस्थिति"],
    "proposed_solution": ["proposed ai solution", "प्रस्तावित ai समाधान"],
    "declaration": ["declaration of originality", "मौलिकता की घोषणा"],
    "drive_url":   ["upload an image with the problem and your proposed solution", "drive link"],
    "consent":     ["consent"],
    "school_name_form": ["please select your school", "school"],
}


def _match_header(headers: list[str], aliases: list[str]) -> int | None:
    """Find the index of the first header that contains every alias token."""
    for i, h in enumerate(headers):
        norm = h.strip().lower()
        if all(token in norm for token in aliases):
            return i
    return None


def _row_get(row: list[str], idx: int | None) -> str | None:
    if idx is None or idx < 0 or idx >= len(row):
        return None
    value = row[idx]
    return value if value and value.strip() else None


def parse_himachal(headers: list[str], row: list[str]) -> ParsedFormResponse | None:
    """Build a ParsedFormResponse from one Himachal-style row, or None if unparseable."""

    def idx(*aliases: str) -> int | None:
        return _match_header(headers, list(aliases))

    find = {
        "timestamp": idx("timestamp"),
        "email": idx("email address"),
        "full_name": idx("write full name in english"),
        "gender": idx("gender"),
        "phone": idx("whatsapp") or idx("mobile"),
        "school_name": idx("school name in english"),
        "class_name": idx("which class"),
        "state": idx("state"),
        "district": idx("district"),
        "language": idx("language"),
        "theme": idx("theme") or idx("थीम"),
        "ai_description": idx("ai description") or idx("ai का सबसे"),
        "situation": idx("situation") or idx("परिस्थिति"),
        "proposed_solution": idx("proposed ai solution") or idx("प्रस्तावित ai"),
        "declaration": idx("declaration") or idx("मौलिकता"),
        "drive_url": idx("upload an image") or idx("drive"),
        "consent": idx("consent"),
        "school_name_form": idx("select your school"),
    }

    def get(key: str) -> str | None:
        return _row_get(row, find[key])

    resources: dict[str, bool] = {}
    for label in HIMACHAL_RESOURCES:
        # each label is its own column with the same prefix
        i = _match_header(headers, [label.lower()])
        resources[label] = _yes(_row_get(row, i))

    full_name = get("full_name")
    school = get("school_name") or get("school_name_form")
    parsed = ParsedFormResponse(
        form_kind="himachal",
        external_ref=_external_ref(0, get("timestamp"), full_name, school),
        timestamp=get("timestamp"),
        email=get("email"),
        full_name=full_name,
        gender=get("gender"),
        phone=get("phone"),
        school_name=school,
        class_name=get("class_name"),
        state=get("state"),
        district=get("district"),
        language=get("language"),
        theme=normalise_theme(get("theme")),
        ai_description=get("ai_description"),
        resources=resources,
        situation=get("situation"),
        proposed_solution=get("proposed_solution"),
        declaration=_yes(get("declaration")),
        consent=_yes(get("consent")),
        school_name_form=get("school_name_form"),
        drive_url=get("drive_url"),
        drive_file_id=_drive_id(get("drive_url")),
        raw={headers[i] if i < len(headers) else f"col{i}": v for i, v in enumerate(row) if v},
    )
    if not (parsed.full_name or parsed.email or parsed.drive_url):
        return None
    return parsed


# ---------------------------------------------------------------------------
# ashramshala (Marathi)
# ---------------------------------------------------------------------------
ASHRAMSHALA_HEADER_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp"],
    "full_name": ["select your name", "निवडा"],
    "division": ["select your division", "select"],
    "project_office": ["select your project office", "select"],
    "ashramshala": ["select your ashramshala", "select"],
    "class_name": ["please select class", "class"],
    "drive_url": ["upload an image"],
    "submission_count": ["count of student submissions"],
    "po_name": ["po name"],
    "school_name_form": ["school name"],
    "theme": ["संकल्पना", "theme"],
    "situation": ["परिस्थिती", "situation"],
    "proposed_solution": ["प्रस्तावित ai उपाय", "proposed ai"],
    "declaration": ["मौलिकतेचे", "declaration"],
}


def parse_ashramshala(headers: list[str], row: list[str]) -> ParsedFormResponse | None:
    def idx(*aliases: str) -> int | None:
        return _match_header(headers, list(aliases))

    find = {
        "timestamp": idx("timestamp"),
        "full_name": idx("select your name"),
        "division": idx("select your division"),
        "project_office": idx("select your project office"),
        "ashramshala": idx("select your ashramshala"),
        "class_name": idx("please select class"),
        "drive_url": idx("upload an image"),
        "submission_count": idx("count of student"),
        "po_name": idx("po name"),
        "school_name_form": idx("school name"),
        "theme": idx("theme") or idx("संकल्पना"),
        "situation": idx("situation") or idx("परिस्थिती"),
        "proposed_solution": idx("proposed ai") or idx("प्रस्तावित ai उपाय"),
        "declaration": idx("declaration") or idx("मौलिकतेचे"),
    }

    def get(key: str) -> str | None:
        return _row_get(row, find[key])

    resources: dict[str, bool] = {}
    for label in ASHRAMSHALA_RESOURCES:
        i = _match_header(headers, [label.lower()])
        resources[label] = _yes(_row_get(row, i))

    school = get("school_name_form")
    full_name = get("full_name")
    parsed = ParsedFormResponse(
        form_kind="ashramshala",
        external_ref=_external_ref(0, get("timestamp"), full_name, school),
        timestamp=get("timestamp"),
        full_name=full_name,
        school_name=school,
        school_name_form=school,
        class_name=get("class_name"),
        division=get("division"),
        project_office=get("project_office"),
        ashramshala=get("ashramshala"),
        resources=resources,
        situation=get("situation"),
        proposed_solution=get("proposed_solution"),
        declaration=_yes(get("declaration")),
        theme=normalise_theme(get("theme")),
        drive_url=get("drive_url"),
        drive_file_id=_drive_id(get("drive_url")),
        po_name=get("po_name"),
        submission_count=int(get("submission_count")) if get("submission_count") else None,
        raw={headers[i] if i < len(headers) else f"col{i}": v for i, v in enumerate(row) if v},
    )
    if not (parsed.full_name or parsed.school_name or parsed.drive_url):
        return None
    return parsed


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------
def detect_kind(headers: list[str]) -> str | None:
    joined = " | ".join(h.lower() for h in headers)
    if "select your ashramshala" in joined or "select your project office" in joined or "select your division" in joined:
        return "ashramshala"
    if "write full name in english" in joined or "अपने प्रोजेक्ट विचार के लिए केवल एक थीम" in joined:
        return "himachal"
    return None


def parse_csv(csv_text: str) -> list[ParsedFormResponse]:
    """Parse a full CSV (any of the two forms). Empty rows and unparseable rows are skipped."""
    reader = csv.reader(io.StringIO(csv_text))
    try:
        headers = next(reader)
    except StopIteration:
        return []
    kind = detect_kind(headers) or "himachal"
    parser = parse_himachal if kind == "himachal" else parse_ashramshala
    results: list[ParsedFormResponse] = []
    for i, row in enumerate(reader, start=1):
        if not row or not any(cell.strip() for cell in row):
            continue
        parsed = parser(headers, row)
        if parsed is None:
            continue
        if not parsed.external_ref:
            parsed.external_ref = f"row-{i:06d}"
        results.append(parsed)
    return results


def parse_csv_file(path: str) -> list[ParsedFormResponse]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return parse_csv(f.read())


def parse_dict_rows(rows: Iterable[dict[str, Any]]) -> list[ParsedFormResponse]:
    """Parse when the input is already a list of dicts (e.g. from the sandbox)."""
    rows = list(rows)
    if not rows:
        return []
    headers = list(rows[0].keys())
    csv_rows: list[list[str]] = []
    for row in rows:
        csv_rows.append([str(row.get(h, "") or "") for h in headers])
    return parse_csv(",".join(headers) + "\n" + "\n".join(",".join(r) for r in csv_rows))