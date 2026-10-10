"""Playwright UI verification + screenshots.

Runs against a live app (http://127.0.0.1:8090 by default), captures real
screenshots into docs/screenshots/, and asserts the UI actually shows data.
Exit code is non-zero if a check fails.

Usage:
    python3 scripts/screenshot.py [--base-url URL] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def shot(page: Page, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    print(f"  [SHOT] {name}.png")


def visit(page: Page, base: str, path: str) -> None:
    response = page.goto(f"{base}{path}", wait_until="networkidle")
    check(response is not None and response.status == 200, f"{path} returns 200")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        page.set_default_timeout(15000)

        print("== Dashboard ==")
        visit(page, base, "/")
        check(page.locator("h1").inner_text() == "Dashboard", "dashboard title")
        check(page.locator(".stat").count() >= 4, "stat cards rendered (>=4)")
        shot(page, out, "dashboard")

        print("== Programs ==")
        visit(page, base, "/programs")
        check(page.locator("h1").inner_text() == "Programs", "programs title")
        check(page.locator("table tbody tr").count() >= 1, "at least one program row")
        shot(page, out, "programs")

        program_link = page.locator("a[href^='/programs/']").first
        if program_link.count() > 0:
            program_href = program_link.get_attribute("href")
            print(f"== Program detail {program_href} ==")
            visit(page, base, program_href)
            check(page.locator("h1").count() == 1, "program detail header")
            shot(page, out, "program_detail")

        print("== Entries ==")
        visit(page, base, "/entries")
        check(page.locator("h1").inner_text() == "Entries", "entries title")
        shot(page, out, "entries")

        print("== Sandbox ==")
        visit(page, base, "/sandbox")
        check(page.locator("h1").inner_text().startswith("Sandbox"), "sandbox title")
        check(page.locator("input[type=file]").count() == 1, "file picker present")
        shot(page, out, "sandbox")

        print("== Settings ==")
        visit(page, base, "/settings")
        check(page.locator("h1").inner_text() == "Settings", "settings title")
        check(page.locator("input[name=reader_mode]").count() == 1, "reader_mode field")
        check(page.locator("input[name=cleaning_mode]").count() == 1, "cleaning_mode field")
        shot(page, out, "settings")

        browser.close()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
