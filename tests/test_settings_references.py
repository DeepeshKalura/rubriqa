"""Every `settings.<name>` in the codebase must resolve to a real Settings field.

Why this file exists. `app/pipeline/readers/minimax_vision.py:73` read
`settings.low_legibility_threshold`. No such field exists. Every page that went
through the vision reader raised `AttributeError` *after* a successful, billable
LLM call, and nothing noticed for a month - because:

  * `make fmt` is `compileall`, which only catches syntax errors, and an
    AttributeError is not one;
  * there is no linter, no typechecker, and no CI;
  * the reader had no test, so nothing ever executed it.

This guard closes that specific hole without adding a dependency. It walks the
AST (so string literals like the template name "settings.html" are correctly
ignored) and asserts each attribute access on a settings object resolves to a
real field, property, or pydantic method.

It also checks `model_copy(update={...})` keys, which pydantic does NOT
validate: a typo in that dict is silently dropped, so a benchmark can quietly
run with the wrong model or the wrong image size and still report a number.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.config import Settings

APP_ROOT = Path("app")
SCRIPTS_ROOT = Path("scripts")

# Attributes that exist on a pydantic BaseModel but are not declared fields.
PYDANTIC_MEMBERS = {
    "model_copy",
    "model_dump",
    "model_dump_json",
    "model_fields",
    "model_fields_set",
    "model_validate",
    "model_json_schema",
}


def _allowed_names() -> set[str]:
    names = set(Settings.model_fields)
    names |= {
        name
        for name, member in vars(Settings).items()
        if isinstance(member, property)
    }
    names |= PYDANTIC_MEMBERS
    return names


ALLOWED = _allowed_names()
# Attribute access is written as `<something>.settings.<name>` or a bare
# `settings.<name>`.
_SETTINGS_BASES = {"settings"}


def _is_settings_base(node: ast.AST) -> bool:
    """True only for the settings object itself: `settings` or `self.settings`.

    Matching is deliberately *immediate*, not recursive. `settings.llm_provider`
    is a field read, but `.lower()` on its result is a method call on a string,
    and a recursive walk would flag it. Only the first hop matters.
    """
    if isinstance(node, ast.Name):
        return node.id in _SETTINGS_BASES
    if isinstance(node, ast.Attribute):
        return node.attr == "settings"
    return False


def _python_files(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(
            path
            for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts
        )
    return out


def _find_bad_attribute_access(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # compileall covers this; stay out of its way
            problems.append(f"{path}:{exc.lineno}: could not parse ({exc.msg})")
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and _is_settings_base(node.value)
                and node.attr not in ALLOWED
            ):
                problems.append(
                    f"{path}:{node.lineno}: `settings.{node.attr}` is not a "
                    f"Settings field, property or pydantic method"
                )
    return problems


def _find_bad_update_keys(files: list[Path]) -> list[str]:
    """`model_copy(update={...})` does not validate keys, so a typo is silent."""
    problems: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "model_copy"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg != "update":
                    continue
                if not isinstance(keyword.value, ast.Dict):
                    continue  # `update=some_var` - nothing static to check
                for key in keyword.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        if key.value not in ALLOWED:
                            problems.append(
                                f"{path}:{node.lineno}: model_copy(update=...) "
                                f"has unknown key '{key.value}' (silently ignored)"
                            )
    return problems


def _py_files() -> list[Path]:
    return _python_files(APP_ROOT, SCRIPTS_ROOT)


def test_settings_attribute_references_all_resolve():
    problems = _find_bad_attribute_access(_py_files())
    assert not problems, "settings attribute references that do not resolve:\n" + "\n".join(
        problems
    )


def test_model_copy_update_keys_are_all_real_fields():
    """A typo in an `update=` dict is dropped without complaint, so a benchmark
    can quietly run against the wrong model or image size."""
    problems = _find_bad_update_keys(_py_files())
    assert not problems, "unknown model_copy(update=...) keys:\n" + "\n".join(problems)


def test_the_guard_would_catch_the_bug_that_started_this():
    """Prove the guard is not vacuous: the exact historical typo must fail."""
    source = "x = self.settings.low_legibility_threshold\n"
    tree = ast.parse(source)
    tree = ast.fix_missing_locations(tree)

    class _FakePath:
        def __init__(self, text: str):
            self._text = text

        def read_text(self, encoding: str = "utf-8", *a, **k) -> str:
            return self._text

    problems = _find_bad_attribute_access([_FakePath(source)])  # type: ignore[list-item]
    assert len(problems) == 1
    assert "low_legibility_threshold" in problems[0]


def test_the_guard_ignores_string_literals():
    """`"settings.html"` is a template name, not an attribute access. A regex
    based check would flag it; an AST walk must not."""
    source = 'template = "settings.html"\n'
    tree = ast.parse(source)
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and _is_settings_base(n.value)
    ]


@pytest.mark.parametrize("field", ["reader_mode", "llm_provider", "gemini_model"])
def test_known_good_references_pass(field):
    source = f"x = settings.{field}\n"
    tree = ast.parse(source)
    bad = [
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and _is_settings_base(n.value) and n.attr not in ALLOWED
    ]
    assert field in ALLOWED
    assert not bad


def test_properties_are_treated_as_valid():
    """`settings.api_key` and `settings.gemini_key` are properties, not
    fields. Without this they would be false positives."""
    assert "api_key" in ALLOWED
    assert "gemini_key" in ALLOWED


def test_scan_covers_the_whole_app_tree():
    files = _py_files()
    assert len(files) >= 40, f"scan found only {len(files)} files - has the tree moved?"
    assert any(f.name == "minimax_vision.py" for f in files)

def test_method_calls_on_field_results_are_not_flagged():
    """`settings.llm_provider.lower()` and `settings.database_url.split('@')` are
    string operations on a real field, not missing settings. A recursive walk
    flagged all three of these in `app/` and would have made the guard
    unusable."""
    source = (
        "a = settings.llm_provider.lower()\n"
        "b = self.settings.database_url.split('@')[-1]\n"
        "c = settings.hybrid_reader_chain.split(',')\n"
    )
    tree = ast.parse(source)
    bad = [
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and _is_settings_base(n.value) and n.attr not in ALLOWED
    ]
    assert not bad


def test_guard_still_catches_a_deep_typo():
    """Immediate-only matching must not have dulled the guard."""
    source = "a = settings.gemini_modle\nb = self.settings.tessaract_langs\n"
    tree = ast.parse(source)
    bad = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and _is_settings_base(n.value) and n.attr not in ALLOWED
    }
    assert bad == {"gemini_modle", "tessaract_langs"}


def test_update_key_checker_flags_unknown_key():
    source = (
        "s = get_settings().model_copy(\n"
        '    update={"llm_provider": "gemini", "vision_max_image_sid": 1400}\n'
        ")\n"
    )

    class _FakePath:
        def __init__(self, text: str):
            self._text = text

        def read_text(self, encoding: str = "utf-8", *a, **k) -> str:
            return self._text

    problems = _find_bad_update_keys([_FakePath(source)])  # type: ignore[list-item]
    assert len(problems) == 1
    assert "vision_max_image_sid" in problems[0]
