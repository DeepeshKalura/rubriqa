"""Form parsers for Himachal and Ashramshala shapes."""
from __future__ import annotations

import csv
import io

import pytest

from app.pipeline.ingestion.forms import (
    ParsedFormResponse,
    detect_kind,
    normalise_theme,
    parse_csv,
    parse_ashramshala,
    parse_dict_rows,
    parse_himachal,
)


HIMACHAL_HEADERS = [
    "Timestamp",
    "Email Address",
    "Write Full Name in English",
    "Gender",
    "WhatsApp / Mobile Number",
    "School Name In English",
    "Which class do you study in?",
    "State",
    "Districts of Himachal Pradesh",
    "Please select your school.",
    "Please select your language",
    "अपने प्रोजेक्ट विचार के लिए केवल एक थीम का चयन करें",
    "AI का सबसे अच्छा वर्णन क्या है?",
    "Resources [Wi-fi]",
    "Resources [Broadband Network]",
    "Resources [Computer]",
    "Resources [Tablet]",
    "Resources [Laptop]",
    "Resources [Mobile]",
    "Situation: Describe the real-life situation",
    "Proposed AI Solution",
    "Declaration of Originality",
    "Upload an image with the problem",
    "Consent",
]

HIMACHAL_ROW = [
    "10/10/2026 9:15:21",
    "aarav@example.com",
    "Aarav Sharma",
    "Male",
    "9876501234",
    "Government Senior Secondary School Hamirpur",
    "Class 7",
    "Himachal Pradesh",
    "Hamirpur",
    "Captain Government Senior Secondary School",
    "Hindi (हिंदी)",
    "स्मार्ट कृषि और ग्रामीण नवाचार में AI",
    "AI is software that learns.",
    "हाँ", "नहीं", "हाँ", "नहीं", "हाँ", "हाँ",
    "किसानों को अक्सर अपनी फसल में होने वाली बीमारी का समय पर पता नहीं चलता।",
    "एक AI ऐप बनाया जा सकता है जिसमें किसान फसल की पत्ती की फोटो लेकर अपलोड करे।",
    "I confirm that the idea shared is my own.",
    "https://drive.google.com/file/d/1abcDEF12345ghIJ/view?usp=sharing",
    "Yes",
]


ASHRAMSHALA_HEADERS = [
    "Timestamp",
    "Select your name",
    "Select your Division",
    "Select your Project Office",
    "Select your Ashramshala",
    "Please select class",
    "Q1 [Wi-fi]",
    "Q1 [Broadband Network]",
    "Q1 [Computer]",
    "Q1 [Smartboard]",
    "Q1 [TV]",
    "Q1 [Projector]",
    "Upload an image with the problem",
    "Count of student submissions",
    "PO Name",
    "School Name",
]

ASHRAMSHALA_ROW = [
    "8/17/2026 20:35:18",
    "कृति देव",
    "नाशिक",
    "यावल",
    "पूरण",
    "Class-6, Class-7, Class-8",
    "Yes", "No", "Yes", "Yes", "Yes", "No",
    "https://drive.google.com/file/d/1UI7K2hahgXBVFxzBXm0I__f9vMaPuvf3/view?usp=drive_link",
    "11",
    "यावल",
    "पूरण",
]


# ---------------------------------------------------------------------------
# detect_kind
# ---------------------------------------------------------------------------
def test_detect_kind_himachal() -> None:
    assert detect_kind(HIMACHAL_HEADERS) == "himachal"


def test_detect_kind_ashramshala() -> None:
    assert detect_kind(ASHRAMSHALA_HEADERS) == "ashramshala"


def test_detect_kind_unknown() -> None:
    assert detect_kind(["foo", "bar"]) is None


# ---------------------------------------------------------------------------
# normalise_theme
# ---------------------------------------------------------------------------
def test_normalise_theme_known() -> None:
    assert normalise_theme("AI for Education") == "ai_education"
    assert normalise_theme("स्मार्ट कृषि और ग्रामीण नवाचार में AI") == "smart_agriculture"
    assert normalise_theme("Health and Wellbeing") == "health_wellbeing"
    assert normalise_theme("AI in Disaster Management") == "disaster_management"


def test_normalise_theme_unknown_slug_fallback() -> None:
    slug = normalise_theme("SOMETHING WEIRD")
    assert slug == "something_weird"


def test_normalise_theme_empty() -> None:
    assert normalise_theme("") is None
    assert normalise_theme(None) is None


# ---------------------------------------------------------------------------
# parse_himachal
# ---------------------------------------------------------------------------
def test_parse_himachal_full_row() -> None:
    parsed = parse_himachal(HIMACHAL_HEADERS, HIMACHAL_ROW)
    assert isinstance(parsed, ParsedFormResponse)
    assert parsed.form_kind == "himachal"
    assert parsed.full_name == "Aarav Sharma"
    assert parsed.gender == "Male"
    assert parsed.phone == "9876501234"
    assert parsed.school_name == "Government Senior Secondary School Hamirpur"
    assert parsed.class_name == "Class 7"
    assert parsed.state == "Himachal Pradesh"
    assert parsed.district == "Hamirpur"
    assert parsed.language == "Hindi (हिंदी)"
    assert parsed.theme == "smart_agriculture"
    assert parsed.drive_url == HIMACHAL_ROW[-2]
    assert parsed.drive_file_id == "1abcDEF12345ghIJ"
    assert parsed.declaration is True
    assert parsed.consent is True
    assert parsed.resources["Wi-fi"] is True
    assert parsed.resources["Broadband Network"] is False
    assert parsed.resources["Computer"] is True


def test_parse_himachal_minimal_returns_none() -> None:
    assert parse_himachal(HIMACHAL_HEADERS, [""] * len(HIMACHAL_HEADERS)) is None


def test_parse_himachal_external_ref_is_stable() -> None:
    a = parse_himachal(HIMACHAL_HEADERS, HIMACHAL_ROW)
    b = parse_himachal(HIMACHAL_HEADERS, HIMACHAL_ROW)
    assert a.external_ref == b.external_ref
    assert len(a.external_ref) > 0


# ---------------------------------------------------------------------------
# parse_ashramshala
# ---------------------------------------------------------------------------
def test_parse_ashramshala_full_row() -> None:
    parsed = parse_ashramshala(ASHRAMSHALA_HEADERS, ASHRAMSHALA_ROW)
    assert parsed.form_kind == "ashramshala"
    assert parsed.full_name == "कृति देव"
    assert parsed.division == "नाशिक"
    assert parsed.project_office == "यावल"
    assert parsed.ashramshala == "पूरण"
    assert parsed.submission_count == 11
    assert parsed.po_name == "यावल"
    assert parsed.school_name_form == "पूरण"
    assert parsed.drive_file_id == "1UI7K2hahgXBVFxzBXm0I__f9vMaPuvf3"
    assert parsed.resources["Wi-fi"] is True
    assert parsed.resources["Smartboard"] is True


def test_parse_ashramshala_drive_id_falls_back_to_query() -> None:
    row = list(ASHRAMSHALA_ROW)
    row[12] = "https://drive.google.com/uc?id=FILE_ID_QUERY"
    parsed = parse_ashramshala(ASHRAMSHALA_HEADERS, row)
    assert parsed.drive_file_id == "FILE_ID_QUERY"


# ---------------------------------------------------------------------------
# parse_csv + parse_dict_rows
# ---------------------------------------------------------------------------
def test_parse_csv_full_himachal() -> None:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(HIMACHAL_HEADERS)
    writer.writerow(HIMACHAL_ROW)
    parsed = parse_csv(buf.getvalue())
    assert len(parsed) == 1
    assert parsed[0].full_name == "Aarav Sharma"


def test_parse_csv_skips_blank_rows() -> None:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(HIMACHAL_HEADERS)
    writer.writerow([""] * len(HIMACHAL_HEADERS))
    writer.writerow(HIMACHAL_ROW)
    parsed = parse_csv(buf.getvalue())
    assert len(parsed) == 1


def test_parse_csv_handles_utf8_bom() -> None:
    text = "\ufeff" + ",".join(HIMACHAL_HEADERS) + "\n" + ",".join(HIMACHAL_ROW)
    parsed = parse_csv(text)
    assert len(parsed) == 1


def test_parse_dict_rows() -> None:
    rows = [
        {
            "Timestamp": "1",
            "Write Full Name in English": "A",
            "Email Address": "a@b.com",
            "State": "HP",
        },
    ]
    parsed = parse_dict_rows(rows)
    assert len(parsed) == 1
    assert parsed[0].full_name == "A"
    assert parsed[0].state == "HP"