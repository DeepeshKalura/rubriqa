"""Capture screenshots of the entry-evaluation mode + sandbox for docs."""
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def main() -> None:
    out = Path("docs/screenshots")
    out.mkdir(parents=True, exist_ok=True)
    base = "http://127.0.0.1:8090"

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        ctx = await browser.new_context(viewport={"width": 1400, "height": 1000})
        page = await ctx.new_page()

        # Programs page
        await page.goto(f"{base}/programs", wait_until="networkidle")
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(out / "programs.png"), full_page=True)
        print("programs.png done")

        # Open the create-program details so the inputs become visible
        await page.evaluate("""
            const d = document.querySelector('details');
            if (d) d.open = true;
        """)
        await page.wait_for_timeout(200)

        # Create a program first
        await page.fill('input[name="name"]', "AI Innovation 2026 — Sandbox demo")
        await page.fill('input[name="rubric_name"]', "Sandbox default rubric")
        await page.fill('input[name="batch_size"]', "250")
        await page.click('button[type="submit"]')
        await page.wait_for_url("**/programs/**", timeout=5000)
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(out / "program_detail.png"), full_page=True)
        print("program_detail.png done")

        # Sandbox page
        await page.goto(f"{base}/sandbox", wait_until="networkidle")
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(out / "sandbox_empty.png"), full_page=True)
        print("sandbox_empty.png done")

        # Pick a bundled sample
        samples = await page.query_selector_all("#samples button")
        if samples:
            await samples[0].click()
            await page.wait_for_timeout(500)

        # Fill the situation + proposed_solution so scoring has content
        await page.fill(
            "#situation",
            "Class 7 students struggle with math word problems because they cannot "
            "understand the language of the question.",
        )
        await page.fill(
            "#proposed_solution",
            "An AI tutor that translates the word problem into simpler Hindi, "
            "explains each step, and gives practice problems of similar type "
            "so students can build confidence and speed.",
        )
        await page.fill(
            "#ai_description",
            "AI is a computer program that can learn patterns from examples and "
            "use those patterns to help solve new problems.",
        )

        # Run comparison
        await page.click("#run-btn")
        # wait for results to render
        await page.wait_for_selector(".paths", timeout=180000)
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(out / "sandbox_compare.png"), full_page=True)
        print("sandbox_compare.png done")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())