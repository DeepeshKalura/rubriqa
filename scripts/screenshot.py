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
        student_cards = page.locator(".stat").count()
        check(student_cards >= 4, "stat cards rendered")
        shot(page, out, "dashboard")

        print("== Classes ==")
        visit(page, base, "/classes")
        check(page.locator(".card").count() >= 1, "class cards rendered")
        class_link = page.locator("a[href^='/classes/']").first
        check(class_link.count() > 0, "class link present")
        shot(page, out, "classes")

        if class_link.count() > 0:
            class_href = class_link.get_attribute("href")
            print(f"== Class detail {class_href} ==")
            visit(page, base, class_href)
            check(page.locator("table").count() >= 1, "batch table rendered")
            shot(page, out, "class_detail")
            batch_link = page.locator("a[href^='/batches/']").first
            if batch_link.count() > 0:
                batch_href = batch_link.get_attribute("href")
                print(f"== Batch {batch_href} ==")
                visit(page, base, batch_href)
                check(page.locator("table tbody tr").count() >= 1, "students listed")
                shot(page, out, "batch_detail")

        print("== Students ==")
        visit(page, base, "/students")
        check(page.locator("table tbody tr").count() >= 1, "student rows")
        shot(page, out, "students")
        # pagination and search
        search = page.locator("input[name=q]")
        if search.count() > 0:
            search.fill("STU00")
            page.wait_for_timeout(700)
            check(page.locator("table tbody tr").count() >= 1, "search returned rows")
            search.fill("")
            page.wait_for_timeout(500)
        student_link = page.locator("a[href^='/students/']").first
        student_href = student_link.get_attribute("href") if student_link.count() else None
        if student_href:
            print(f"== Student {student_href} ==")
            visit(page, base, student_href)
            check(page.locator("h1").count() == 1, "student header")
            shot(page, out, "student_detail")

        print("== Exams ==")
        visit(page, base, "/exams")
        check(page.locator("table tbody tr").count() >= 1, "exam rows")
        shot(page, out, "exams")
        exam_link = page.locator("a[href^='/exams/']").first
        exam_href = exam_link.get_attribute("href") if exam_link.count() else None
        if exam_href:
            visit(page, base, exam_href)
            check(page.locator("table tbody tr").count() >= 1, "question rows")
            shot(page, out, "exam_detail")
            results_href = f"{exam_href}/results"
            print(f"== Results {results_href} ==")
            visit(page, base, results_href)
            check(page.locator("table tbody tr").count() >= 1, "result rows")
            shot(page, out, "exam_results")

        print("== Submission detail ==")
        submission_link = page.locator("a[href^='/submissions/']").first
        if submission_link.count() == 0:
            visit(page, base, "/results")
            submission_link = page.locator("a[href^='/submissions/']").first
        if submission_link.count() > 0:
            sub_href = submission_link.get_attribute("href")
            visit(page, base, sub_href)
            check(page.locator("h1").count() == 1, "submission header")
            shot(page, out, "submission_detail")

        print("== Jobs ==")
        visit(page, base, "/jobs")
        check(page.locator("#jobs-table").count() == 1, "jobs table present")
        shot(page, out, "jobs")
        job_link = page.locator("a[href^='/jobs/']").first
        if job_link.count() > 0:
            job_href = job_link.get_attribute("href")
            visit(page, base, job_href)
            check(page.locator("h1").count() == 1, "job detail header")
            check(page.locator("#job-body").count() == 1, "job body (live polling)")
            shot(page, out, "job_detail")

        print("== Review ==")
        visit(page, base, "/review")
        check(page.locator("h1").inner_text() == "Review", "review page")
        shot(page, out, "review")

        print("== Settings ==")
        visit(page, base, "/settings")
        check(page.locator("form").count() >= 1, "settings form")
        shot(page, out, "settings")

        print("== Import ==")
        visit(page, base, "/imports")
        check(page.locator("h1").inner_text() == "Import", "imports page")
        shot(page, out, "imports")
        visit(page, base, "/imports/new")
        check(page.locator("form").count() >= 2, "import forms")
        shot(page, out, "import_new")

        print("== Google Forms ==")
        visit(page, base, "/google")
        check(page.locator("h1").inner_text() == "Google Forms", "google page")
        shot(page, out, "google_forms")

        print("== Health ==")
        visit(page, base, "/healthz")
        check("ok" in page.content(), "healthz ok")

        browser.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("All UI checks passed. Screenshots in", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
