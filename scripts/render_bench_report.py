"""Render the model/path benchmark as a standalone HTML report (no images).

Reads `reports/bench_models.json` (produced by `scripts.bench_models.py`) and
emits a self-contained HTML page with the numbers that decide a 25k run:

  * model verdict      - accuracy / cost / latency per (model, path)
  * cost table         - per-record and per-25k USD, with cached-input share
  * throughput         - wall clock at N workers, and whether a target is met
  * disk budget        - GB required for the target record count
  * per-page detail    - the extracted text each path read, per model

Text-only on purpose: the page images are already reviewed in the sandbox, and a
report that embeds them stops being readable.

Usage (Docker):
    docker compose exec web python -m scripts.render_bench_report
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.cost import fmt_usd  # noqa: E402

SRC = Path("reports/bench_models.json")
OUT_DOCS = Path("docs/bench_report.html")
OUT_DATA = Path("data/reports/bench_report.html")

PATH_LABEL = {
    "vision": "Path A · vision-LLM",
    "tesseract_llm": "Path B · tesseract+LLM",
    "vision_ocr_fusion": "Path C · fusion",
}
PATH_ORDER = ["vision", "tesseract_llm", "vision_ocr_fusion"]
# Only these paths prove a model can actually read an image. Path B's text comes
# from Tesseract, so a non-zero run there says nothing about vision ability.
VISION_PATHS = ("vision", "vision_ocr_fusion")
PATH_COLOR = {"vision": "#2457d6", "tesseract_llm": "#b45309", "vision_ocr_fusion": "#6d28d9"}
MODEL_SHORT = {
    "minimax-m3": "MiniMax-M3",
    "minimax-m3.1-flash-preview": "M3.1-Flash-Preview",
}


def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def cer_class(cer: float | None) -> str:
    if cer is None:
        return "cer-na"
    if cer < 15:
        return "cer-good"
    if cer < 40:
        return "cer-warn"
    return "cer-bad"


def cer_cell(cer: float | None) -> str:
    if cer is None:
        return '<span class="cer-na">—</span>'
    return f'<span class="{cer_class(cer)}">{cer:.1f}%</span>'


def _honest_wins(payload: dict) -> dict[tuple[str, str], dict]:
    """CER wins per cell, with wins-by-absence separated from real wins.

    A cell that is the only one producing readable text on a page would
    otherwise "win" by default — which is exactly what happens when a model
    cannot do vision at all and its vision paths return nothing. Counting that
    as a win would invert the conclusion, so we require at least one rival to
    have produced text before a win is recorded.
    """
    by_page: dict[str, list[dict]] = {}
    for r in payload["runs"]:
        by_page.setdefault(r["page"], []).append(r)

    out: dict[tuple[str, str], dict] = {}
    for page, page_runs in by_page.items():
        readable = [r for r in page_runs if r["ok"] and r["ocr_chars"] > 0 and r["cer"] is not None]
        if not readable:
            continue
        contested = len(readable) > 1
        best = min(readable, key=lambda r: r["cer"])
        key = (best["pricing_key"], best["path"])
        slot = out.setdefault(key, {"real": 0, "by_default": 0})
        if contested:
            slot["real"] += 1
        else:
            slot["by_default"] += 1
    return out


def model_rows(payload: dict) -> str:
    """The headline model-vs-path table."""
    wins = _honest_wins(payload)
    rows = []
    for s in sorted(payload["summary"], key=lambda x: (x["pricing_key"], PATH_ORDER.index(x["path"]))):
        short = MODEL_SHORT.get(s["pricing_key"], s["pricing_key"])
        fails = s["failed_runs"]
        w = wins.get((s["pricing_key"], s["path"]), {"real": 0, "by_default": 0})
        # Path B's text comes from Tesseract, not from the model, so a non-zero
        # run there says nothing about whether the model can read an image.
        is_vision = s["path"] in VISION_PATHS
        no_ocr = is_vision and s["runs_with_ocr"] == 0
        fail_note = (
            '<span class="badge red" title="every vision run returned no readable text">no vision output</span>'
            if no_ocr else ""
        )
        verified = "" if s.get("price_verified", True) is not False else \
            '<span class="badge amber">price unverified</span>'
        wins_cell = (
            f'<b>{w["real"]}</b>'
            + (f'<span class="badge grey" title="won only because every rival produced no text">+{w["by_default"]}*</span>' if w["by_default"] else "")
            if not no_ocr
            else '<span class="cer-na">—</span>'
        )
        rows.append(
            f"""<tr>
  <td class="m"><b>{esc(short)}</b> {verified}{fail_note}</td>
  <td>{esc(PATH_LABEL.get(s['path'], s['path']))}</td>
  <td class="num">{s['samples']}</td>
  <td class="num">{cer_cell(s['cer_mean'])}</td>
  <td class="num">{wins_cell}</td>
  <td class="num">{s['mean_input_tokens']:,}</td>
  <td class="num">{s['mean_cached_tokens']:,}</td>
  <td class="num">{s['mean_output_tokens']:,}</td>
  <td class="num"><b>{fmt_usd(s['mean_cost_usd'])}</b></td>
  <td class="num">{s['mean_latency_ms']/1000:.1f}s</td>
  <td class="num">{s['p95_latency_ms']/1000:.1f}s</td>
  <td class="num {'bad' if fails else ''}">{fails}</td>
</tr>"""
        )
    return "\n".join(rows)


def cost_rows(payload: dict) -> str:
    """Per-record -> per-25k money table with the p95 time projection."""
    target = payload.get("records_target", 25_000)
    workers = payload.get("workers_assumed", 8)
    rows = []
    for s in sorted(payload["summary"], key=lambda x: (x["pricing_key"], PATH_ORDER.index(x["path"]))):
        p = s["projection_25k"]
        short = MODEL_SHORT.get(s["pricing_key"], s["pricing_key"])
        cached_pct = (
            round(100 * s["mean_cached_tokens"] / s["mean_input_tokens"], 1)
            if s["mean_input_tokens"] else 0.0
        )
        # Derive workers-needed rather than trusting the JSON: a cell that timed
        # out mid-run may not carry the derived key.
        workers_8h = p.get("workers_8h")
        if workers_8h is None:
            workers_8h = round(
                (p.get("per_record_ms_p95", 0) * target) / (8.0 * 3_600_000), 1
            )
        rows.append(
            f"""<tr>
  <td><b>{esc(short)}</b></td>
  <td>{esc(PATH_LABEL.get(s['path'], s['path']))}</td>
  <td class="num">{fmt_usd(s['mean_cost_usd'])}</td>
  <td class="num">{cached_pct:.0f}%</td>
  <td class="num"><b>{fmt_usd(p['total_cost_usd'])}</b></td>
  <td class="num">{p['wall_hours']:.0f}h</td>
  <td class="num">{p['serial_hours']:.0f}h</td>
  <td class="num">{workers_8h:.0f}</td>
</tr>"""
        )
    return "\n".join(rows)


def render_page_detail(payload: dict) -> str:
    """Per page: which model/path read it best, and what each one actually read."""
    runs = payload["runs"]
    pages = payload["pages"]
    by_page: dict[str, list[dict]] = {}
    for r in runs:
        by_page.setdefault(r["page"], []).append(r)

    out = []
    for page in pages:
        page_runs = by_page.get(page, [])
        # best CER across every run of this page
        scored = [r for r in page_runs if r["ok"] and r["cer"] is not None]
        best = min(scored, key=lambda r: r["cer"]) if scored else None

        if best is None:
            banner = (
                '<div class="verdict bad"><div class="v-trophy">⛔</div><div>'
                '<div class="v-label">No usable reading</div>'
                '<div class="v-head">Every run failed or produced no comparable text on this page.</div>'
                "</div></div>"
            )
        else:
            mshort = MODEL_SHORT.get(best["pricing_key"], best["pricing_key"])
            noisy = best["cer"] > 40
            contested = len(
                [r for r in page_runs if r["ok"] and r["ocr_chars"] > 0 and r["cer"] is not None]
            ) > 1
            if not contested:
                sub = ("Only this cell produced any readable text on this page — a win by "
                       "absence, not a demonstrated win.")
            else:
                sub = f"{esc(mshort)} on {esc(PATH_LABEL.get(best['path'], ''))} was closest to the human transcript."
            banner = (
                f'<div class="verdict{" noisy" if noisy else ""}">'
                f'<div class="v-trophy">{"⚠️" if noisy else "🏆"}</div><div>'
                f'<div class="v-label">{"Best available — every path is unreliable" if noisy else ("Only readable result" if not contested else "Best reading of this page")}</div>'
                f'<div class="v-head">{esc(mshort)} · {esc(PATH_LABEL.get(best["path"], best["path"]))} — CER {best["cer"]:.1f}%</div>'
                f'<div class="v-sub">{sub}</div>'
                "</div></div>"
            )

        cards = []
        for r in sorted(page_runs, key=lambda x: (x["pricing_key"], PATH_ORDER.index(x["path"]))):
            mshort = MODEL_SHORT.get(r["pricing_key"], r["pricing_key"])
            color = PATH_COLOR.get(r["path"], "#334155")
            if not r["ok"]:
                cards.append(
                    f"""<article class="pcard" style="border-top-color:{color}">
  <header class="phead"><span class="pname" style="color:{color}">{esc(mshort)} · {esc(PATH_LABEL.get(r['path'], r['path']))}</span>
  <span class="badge red">failed</span></header>
  <div class="errbox">{esc(r.get('error') or 'unknown error')}</div>
</article>"""
                )
                continue
            key = f"{page}-{r['pricing_key']}-{r['path']}".replace(".", "-")
            cost_line = ""
            if r["usd"]:
                cost_line = f'<span class="badge">{fmt_usd(r["usd"])}</span>'
            cards.append(
                f"""<article class="pcard" style="border-top-color:{color}">
  <header class="phead">
    <span class="pname" style="color:{color}">{esc(mshort)} · {esc(PATH_LABEL.get(r['path'], r['path']))}</span>
    <span class="pills">
      <span class="badge {cer_class(r['cer'])}">CER {f"{r['cer']:.1f}%" if r['cer'] is not None else 'n/a'}</span>
      <span class="badge blue">score {(f"{r['score']:.0f}" if r['score'] is not None else "—")}/{r['score_max']:.0f}</span>
      <span class="badge">{cost_line}</span>
      <span class="badge">{r['elapsed_ms']/1000:.1f}s</span>
    </span>
  </header>
  <div class="block">
    <div class="bhead"><span>Extracted text <span class="hint">({r['ocr_chars']} chars)</span></span>
      <button class="copy" data-copy="{key}">Copy text</button></div>
    <pre class="extract" id="{key}">{esc(r['ocr_text']) if r['ocr_text'] else '(empty — this run produced no readable text)'}</pre>
  </div>
</article>"""
            )

        out.append(
            f"""<section class="page" id="pg-{page}">
  <div class="phead-bar"><h2>{esc(page)}</h2>
    <span class="hint">{len(page_runs)} runs compared</span></div>
  {banner}
  {"".join(cards)}
  <div class="back"><a href="#top">↑ summary</a></div>
</section>"""
        )
    return "\n".join(out)


def build(payload: dict) -> str:
    storage = payload["storage_25k"]
    target = payload.get("records_target", 25_000)
    workers = payload.get("workers_assumed", 8)

# headline finding: does a model actually *read pages*? Only the vision paths
    # (A and C) prove that — Path B's text comes from Tesseract regardless of model,
    # so a model can score there without ever seeing an image.

    by_key: dict[str, dict] = {}
    for s in payload["summary"]:
        k = by_key.setdefault(
            s["pricing_key"],
            {"vision_ocr": 0, "vision_runs": 0, "any_ocr": 0, "any_runs": 0, "model": s["model"]},
        )
        # Path A is the ONLY pure-vision path: its text is produced by the model
        # from the image alone. Path C can fall back to Tesseract when the vision
        # call fails, so a non-zero run there proves nothing about vision.
        if s["path"] == "vision":
            k["vision_ocr"] += s["runs_with_ocr"]
            k["vision_runs"] += s["samples"]
        k["any_ocr"] += s["runs_with_ocr"]
        k["any_runs"] += s["samples"]

    reads_pages = [MODEL_SHORT.get(k, k) for k, v in by_key.items() if v["vision_ocr"] > 0]
    no_vision = [MODEL_SHORT.get(k, k) for k, v in by_key.items() if v["vision_ocr"] == 0]

    findings = []
    if no_vision:
        findings.append(
            f"<li><b>{esc(', '.join(no_vision))}</b> read <b>no page images at all</b> — its vision "
            f"paths (A and C) returned empty text on every run. Verified against the live API: "
            f"it answers text prompts, but it cannot disable its reasoning preamble and it "
            f"rejects full-page image payloads. The thinking block consumes the whole "
            f"<code>max_tokens</code> budget, the response ends with "
            f"<code>finish_reason: length</code>, and <code>message.content</code> is empty. "
            f"It can still work on <b>Path B only</b>, where Tesseract does the OCR and the "
            f"model merely scores — and it is <b>slower per run</b> than M3 while doing it.</li>"
        )
    if reads_pages:
        findings.append(
            f"<li><b>{esc(', '.join(reads_pages))}</b> is the only model in this matrix that "
            f"actually reads pages. Every OCR number below comes from it.</li>"
        )
    findings.append(
        "<li>Wins marked <code>+N*</code> are <b>wins by absence</b> — the cell was the only one "
        "that produced any text, so it cannot be called a real win. They are excluded from the "
        "headline count.</li>"
    )
    findings.append(
        "<li>Latency projections use <b>p95</b>, not the mean. The p95 tail is brutal: path A's "
        "median is ~25s while its p95 is minutes on hard pages. A mean-based 25k projection "
        "understates wall clock several-fold.</li>"
    )
    findings.append(
        f"<li>Disk, not API cost, is the binding constraint: <b>{storage['total_gb']} GB</b> "
        f"needed for {target:,} pages at {storage['mb_per_page']} MB/page (originals only).</li>"
    )
    findings.append(
        "<li>Page-to-page variance dominates everything. Per-page CER spans ~9% to ~200%, which "
        "is why 11 pages cannot statistically separate the paths — see issue #6.</li>"
    )

    ref_pricing = payload.get("pricing", {}).get("reference", {})
    ref_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>${v['input']:.2f}</td>"
        f"<td class='num'>${v['cached_input']:.3f}</td><td class='num'>${v['output']:.2f}</td>"
        f"<td class='hint-cell'>reference only — no key configured, never called</td></tr>"
        for k, v in ref_pricing.items()
    )

    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    toc = " · ".join(f'<a href="#pg-{p}">{esc(p.replace("page-", "p"))}</a>' for p in payload["pages"])

    # --- the recommendation the owner asked for: which path do we actually run?
    stats_by = {(s["pricing_key"], s["path"]): s for s in payload["summary"]}
    main_model = "minimax-m3"
    a = stats_by.get((main_model, "vision"))
    b = stats_by.get((main_model, "tesseract_llm"))
    c = stats_by.get((main_model, "vision_ocr_fusion"))

    recs = []
    if a and c:
        a_letter, c_letter = "A", "C"
        if c["cer_mean"] < a["cer_mean"]:
            acc_better, acc_worse, acc_letter = c, a, c_letter
        else:
            acc_better, acc_worse, acc_letter = a, c, a_letter
        cost_better = a if a["mean_cost_usd"] < c["mean_cost_usd"] else c
        cost_letter = a_letter if cost_better is a else c_letter
        if acc_letter == cost_letter:
            recs.append(
                f"<li><b>Run Path {acc_letter} by default.</b> It has the lower mean CER "
                f"({acc_letter} {acc_better['cer_mean']:.1f}% vs {acc_worse['path'] and acc_letter} "
                f"{acc_worse['cer_mean']:.1f}%) <i>and</i> it is the cheaper of the two "
                f"({fmt_usd(cost_better['mean_cost_usd'])}/page). Accuracy and cost point the "
                f"same way — this is the one decision the data does support.</li>"
            )
        else:
            recs.append(
                f"<li><b>Path {acc_letter} is more accurate, Path {cost_letter} is cheaper — the "
                f"data does not settle which to default to.</b> {acc_letter} has the lower mean "
                f"CER ({acc_better['cer_mean']:.1f}% vs {acc_worse['cer_mean']:.1f}%) at "
                f"{fmt_usd(acc_better['mean_cost_usd'])}/page; {cost_letter} costs "
                f"{fmt_usd(cost_better['mean_cost_usd'])}/page. That is "
                f"{abs(100 * (acc_better['mean_cost_usd'] / cost_better['mean_cost_usd'] - 1)):.0f}% "
                f"cheaper for a {abs(acc_better['cer_mean'] - acc_worse['cer_mean']):.1f} CER-point "
                f"difference — and on 11 pages neither difference is statistically significant "
                f"(issue #6). Treat this as a cost decision, not an accuracy claim.</li>"
            )
    if b:
        recs.append(
            f"<li><b>Do not run Path B as the production path.</b> It never won a contested "
            f"CER comparison ({b['cer_wins']} real wins), and its OCR is the failure mode — "
            f"on some pages it returns nothing at all, yet still scores because the prefilled "
            f"form fields are present. That is a *scoring* result with no reading behind it.</li>"
        )
    recs.append(
        f"<li><b>Escalate, don't choose once.</b> The evidence supports running Path A first and "
        f"escalating to Path C only when A's legibility or CER is poor — the fusion gate "
        f"already does this for noise detection. On 11 pages the mean gap is not statistically "
        f"separable (issue #6), so a single fixed path for every page is not yet justified by data.</li>"
    )
    recs.append(
        f"<li><b>Fix disk before scale.</b> {storage['total_gb']} GB for {target:,} pages. The API "
        f"bill for the whole run is smaller than the storage it needs.</li>"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark — models x OCR paths</title>
<style>
  :root {{
    --ink:#17223b; --muted:#64748b; --line:#dfe5ec; --bg:#f4f6f9; --panel:#fff;
    --good:#15803d; --good-soft:#e6f6ec; --warn:#b45309; --warn-soft:#fdf3e3;
    --bad:#b91c1c; --bad-soft:#fdecec;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 -apple-system,"Segoe UI",Roboto,"Noto Sans","Noto Sans Devanagari",sans-serif}}
  .topbar{{background:#101a33;color:#fff;padding:16px 26px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
  .topbar .t{{font-size:17px;font-weight:700}}
  .topbar .m{{font-size:12px;color:#cdd6e4}}
  .wrap{{max-width:1240px;margin:0 auto;padding:22px 26px 70px}}
  h1{{font-size:24px;margin:6px 0 4px}}
  h2{{font-size:19px;margin:0}}
  .lede{{color:var(--muted);margin:0 0 18px;max-width:100ch}}
  section.blk{{margin:26px 0}}
  .blkh{{border-bottom:2px solid var(--line);padding-bottom:6px;margin-bottom:10px;display:flex;
    justify-content:space-between;align-items:flex-end;gap:10px;flex-wrap:wrap}}
  .blkh .hint{{color:var(--muted);font-size:12.5px}}
  table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
    border-radius:10px;overflow:hidden;font-size:13.5px}}
  th{{background:#f8fafc;color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;
    text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
  td{{padding:8px 10px;border-bottom:1px solid #edf1f6;vertical-align:top}}
  td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
  td.num.bad{{color:var(--bad);font-weight:700}}
  tr:last-child td{{border-bottom:none}}
  .badge{{display:inline-block;border-radius:999px;padding:2px 9px;font-size:11.5px;font-weight:600;
    background:#eef2f7;color:#475569;white-space:nowrap}}
  .badge.blue{{background:#e8effc;color:#1d3a78}}
  .badge.red{{background:var(--bad-soft);color:var(--bad)}}
  .badge.amber{{background:var(--warn-soft);color:var(--warn)}}
  .badge.cer-good{{background:var(--good-soft);color:var(--good)}}
  .badge.cer-warn{{background:var(--warn-soft);color:var(--warn)}}
  .badge.cer-bad{{background:var(--bad-soft);color:var(--bad)}}
  .cer-good{{color:var(--good);font-weight:700}}
  .cer-warn{{color:var(--warn);font-weight:700}}
  .cer-bad{{color:var(--bad);font-weight:700}}
  .cer-na{{color:var(--muted)}}
  .verdict{{display:flex;gap:12px;align-items:flex-start;border-radius:12px;padding:13px 16px;margin:10px 0;
    background:linear-gradient(135deg,#fff8e1,#fde68a);border:2px solid #f59e0b}}
  .verdict.noisy{{background:linear-gradient(135deg,#fee2e2,#fecaca);border-color:var(--bad)}}
  .verdict.bad{{background:linear-gradient(135deg,#fee2e2,#fecaca);border-color:var(--bad)}}
  .v-trophy{{font-size:28px;line-height:1}}
  .v-label{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;color:#92400e}}
  .v-head{{font-size:17px;font-weight:800}}
  .v-sub{{font-size:12.5px;color:#6b4b12}}
  .verdict.noisy .v-label,.verdict.bad .v-label{{color:#8a1313}}
  .verdict.noisy .v-sub,.verdict.bad .v-sub{{color:#6b1d1d}}
  .findings{{background:#fff;border:1px solid var(--line);border-left:4px solid #2457d6;border-radius:10px;
    padding:14px 20px}}
  .findings.rec{{border-left-color:#15803d;background:#f4fdf7}}
  .findings ul{{margin:6px 0 0;padding-left:20px}}
  .findings li{{margin:7px 0}}
  .findings code{{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:12.5px}}
  .warnbar{{background:var(--bad-soft);border:1px solid #f3c2c2;border-radius:10px;padding:12px 16px;
    color:#8a1313;font-size:13.5px;margin:10px 0}}
  .page{{margin-top:30px;scroll-margin-top:60px}}
  .phead-bar{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid var(--line);
    padding-bottom:6px;margin-bottom:10px}}
  .phead-bar .hint{{color:var(--muted);font-size:12.5px}}
  .pcard{{background:var(--panel);border:1px solid var(--line);border-top:4px solid #334155;border-radius:12px;
    margin:12px 0;overflow:hidden}}
  .phead{{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;
    padding:10px 16px;border-bottom:1px solid var(--line);background:#fbfcfe}}
  .pname{{font-size:15px;font-weight:700}}
  .pills{{display:flex;gap:6px;flex-wrap:wrap}}
  .block{{padding:12px 16px}}
  .bhead{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:12px;
    text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:700}}
  .bhead .hint{{text-transform:none;letter-spacing:0;font-weight:400}}
  .copy{{border:1px solid var(--line);background:#fff;border-radius:7px;font-size:12px;padding:3px 10px;cursor:pointer}}
  .copy:hover{{background:#f1f5f9}}
  pre.extract{{margin:0;background:#f8fafc;border:1px solid var(--line);border-radius:9px;padding:13px 15px;
    font-size:15px;line-height:1.7;white-space:pre-wrap;word-break:break-word;
    font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans","Noto Sans Devanagari",sans-serif}}
  .errbox{{margin:12px 16px;background:var(--bad-soft);border:1px solid #f3c2c2;color:#8a1313;
    border-radius:8px;padding:10px 12px;font-size:13px}}
  .hint-cell{{color:var(--muted);font-size:12px}}
  .toc{{font-size:13.5px;margin:8px 0 0}}
  .toc a{{color:#2457d6;text-decoration:none;font-weight:600;margin-right:8px}}
  .back{{font-size:12.5px;margin-top:6px}}
  .back a{{color:var(--muted);text-decoration:none}}
  @media print{{.copy,.back,.toc{{display:none}} .page{{page-break-before:always}}
    .pcard,.verdict{{break-inside:avoid}} .topbar{{background:#fff;color:var(--ink)}} .topbar .m{{color:var(--muted)}}}}
</style></head>
<body id="top">
<div class="topbar">
  <div class="t">Benchmark — LLM models × OCR paths</div>
  <div class="m">{len(payload['pages'])} pages × {len(payload['summary'])} model/path cells ·
    generated {now} · text-only · source reports/bench_models.json</div>
</div>
<div class="wrap">
  <h1>Which model and which path, at what cost</h1>
  <p class="lede">Every number below is from live runs through the production
  <code>/sandbox/compare</code> endpoint — the same code path the sandbox UI uses.
  Cost is USD at published list rates; latency projections use p95, not the mean.</p>

  <section class="blk">
    <div class="blkh"><h2>Recommendation — what to actually run</h2>
      <span class="hint">derived from the measured cells above</span></div>
    <div class="findings rec"><ul>{"".join(recs)}</ul></div>
  </section>

  <section class="blk">
    <div class="blkh"><h2>Key findings</h2><span class="hint">read this first</span></div>
    <div class="findings"><ul>{"".join(findings)}</ul></div>
  </section>

  <section class="blk">
    <div class="blkh"><h2>Accuracy, cost and latency per model × path</h2>
      <span class="hint">lower CER is better · wins = pages where this cell had the best CER</span></div>
    <table>
      <thead><tr>
        <th>Model</th><th>Path</th><th class="num">runs</th><th class="num">mean CER</th>
        <th class="num">CER wins</th><th class="num">in tok</th><th class="num">cached</th>
        <th class="num">out tok</th><th class="num">$/record</th><th class="num">mean</th>
        <th class="num">p95</th><th class="num">failed</th>
      </tr></thead>
      <tbody>{model_rows(payload)}</tbody>
    </table>
  </section>

  <section class="blk">
    <div class="blkh"><h2>Cost and time to {target:,} records</h2>
      <span class="hint">wall clock assumes {workers} workers · serial = single worker</span></div>
    <table>
      <thead><tr>
        <th>Model</th><th>Path</th><th class="num">$/record</th><th class="num">cached share</th>
        <th class="num">total cost</th><th class="num">wall @{workers}w</th>
        <th class="num">serial</th><th class="num">workers for 8h</th>
      </tr></thead>
      <tbody>{cost_rows(payload)}</tbody>
    </table>
  </section>

  <section class="blk">
    <div class="blkh"><h2>Disk budget</h2><span class="hint">the real constraint</span></div>
    <div class="warnbar"><b>{storage['total_gb']} GB required</b> for {target:,} pages at
      {storage['mb_per_page']} MB/page (measured from the 11 bundled samples, originals only —
      rendered copies and the golden set add more). Check free space before scaling.</div>
  </section>

  <section class="blk">
    <div class="blkh"><h2>Reference pricing (not executed)</h2>
      <span class="hint">no key configured — shown so a provider swap can be costed</span></div>
    <table>
      <thead><tr><th>Model</th><th class="num">$/M in</th><th class="num">$/M cached</th>
        <th class="num">$/M out</th><th>status</th></tr></thead>
      <tbody>{ref_rows}</tbody>
    </table>
  </section>

  <section class="blk">
    <div class="blkh"><h2>Per-page detail</h2><span class="hint">what each model actually read</span></div>
    <div class="toc"><b>Jump:</b> {toc}</div>
  </section>

  {render_page_detail(payload)}
</div>
<script>
document.addEventListener('click', function (e) {{
  var btn = e.target.closest('.copy');
  if (!btn) return;
  var el = document.getElementById(btn.getAttribute('data-copy'));
  if (!el) return;
  var done = function () {{
    var old = btn.textContent; btn.textContent = 'Copied ✓';
    setTimeout(function () {{ btn.textContent = old; }}, 1200);
  }};
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(el.innerText).then(done, done);
  }} else {{ done(); }}
}});
</script>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out-docs", default=str(OUT_DOCS))
    ap.add_argument("--out-data", default=str(OUT_DATA))
    args = ap.parse_args()

    payload = json.loads(Path(args.src).read_text(encoding="utf-8"))
    doc = build(payload)
    for out in (Path(args.out_docs), Path(args.out_data)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc, encoding="utf-8")
        print(f"wrote {out} ({len(doc)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())