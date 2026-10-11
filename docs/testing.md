# Human testing guide

This is the runbook a staff member follows end-to-end on a clean machine.
Read it straight through once, then use the per-flow checklists when you
sit down to test.

Every step uses `make`. Never run Python on the host. Never paste a key
into a chat or commit. Keys come from `.env`.

---

## 0. One-time setup

```bash
# 1. start the stack
cp .env.example .env
# open .env, fill MINIMAX_API_KEY (and TESSERACT_LANGS=eng+hin+mar if you test Marathi)
make up
make migrate

# 2. seed a demo program + a few sample entries (with real local images)
make seed
```

Open <http://127.0.0.1:8090>. You should see the Dashboard with counts.

**Why each step exists**

- `make up` boots `web` + `postgres` in Docker; no host Python touched.
- `make migrate` applies Alembic (the 13 tables for the entry-evaluation mode).
- `make seed` inserts one Program, one Rubric, three entries that point at
  files under `data/samples/`.

---

## 1. The four flows you must test

| # | Flow | URL | What it proves |
|---|---|---|---|
| F1 | Program + entry list | `/programs/{id}` | Pagination, form answers, drive metadata display |
| F2 | Entry detail + scores | `/entries/{id}` | OCR original vs cleaned, per-criterion scores, human override |
| F3 | Sandbox (3-way OCR compare) | `/sandbox` | Vision / Tesseract+LLM / Vision+OCR fusion on the same image, with per-page golden transcript + CER |
| F4 | Batch + review samples | `/programs/{id}/batches/{bid}` | Random/AI-flagged/low-confidence sample mix, batch state machine |

If F1–F4 work, the entry-evaluation critical path works.

---

## 2. Flow F1 — Program + entries

1. Dashboard → Programs → click a program.
2. Confirm the entry table paginates (no console errors, no row overflow).
3. Spot-check three entries: name, district, theme, score column, state.
4. Change the page-size selector (25/50/100). Reload. Same first row, no duplicates.

**Look right?** table is paginated, header row visible at every scroll position,
state chips use colour consistently (pending vs approved vs rejected), no
horizontal scroll on a 1280px window.

---

## 3. Flow F2 — Entry detail + scoring

1. Click any entry row.
2. Read the **OCR original** and the **cleaned text** side by side. They must
   be different when the cleaning pass made changes; identical when it didn't.
3. Read the **score card**: per-criterion bars, strengths, gaps, observations.
4. **Override the score** in the human override form. Save. Reload the page.
   The entry must now show your note + corrected score.
5. Flip `status` from `pending` to one of `approved` / `corrected` / `rejected`.
   Reload. The state chip must persist.

**Look right?** each criterion has a bar whose width matches its value, the
human override survives a refresh, the state chip is colour-coded.

---

## 4. Flow F3 — Sandbox (the most important test)

1. Go to `/sandbox`.
2. Drag in any image from `data/samples/pages/` (Marathi handwriting, printed
   Hindi, scanned English — pick at least one of each).
3. Click **Compare paths**.
4. The **Path A** (vision-LLM), **Path B** (Tesseract+LLM), and **Path C**
   (vision+OCR fusion) panels should all fill in. Each card shows the OCR
   text, the score, the CER pill (when the uploaded image matches a bundled
   sample with a golden transcript in `data/samples/golden/`), and a small
   status badge.
5. Open the **Path C** `<details>` to inspect its two component transcripts
   (vision + tesseract) before the fusion LLM merged them.
6. Read the bottom **side-by-side diff** table. Look at:
   - total score delta between paths
   - token usage delta (vision is heavier; fusion is heaviest)
   - elapsed time delta
   - CER pill colours: green `< 15`, yellow `< 40`, red `>= 40`
7. If a Path C card shows the yellow **Gate** badge
   (`tesseract too long, used vision`), the fusion LLM call was skipped
   because tesseract's text was suspiciously long vs vision's — the result
   is the vision text. This is the `fusion_max_tess_to_vision_ratio` flag
   (default 1.5). See `docs/sandbox/fusion_gate.md`.
8. Click **Reset sandbox**. Confirm entries vanish.

**Look right?** all three paths produced non-empty text (or an honest
red error card), totals differ, no column ever shows `error: <stack
trace>`, and CER pills appear on bundled-sample uploads.

If a path returns an `error:` card saying *"OCR produced no text and the
form fields are empty — nothing to score."* that's the correct honest
outcome (the `nothing_to_score` gate). See `docs/limitations.md #3`.

---

## 5. Flow F4 — Batch + review samples

1. On a Program page, click **Start batch**. Pick a small section (≤ 30 entries
   first time).
2. Watch the batch row: `pending → processing → awaiting_review → completed`.
3. Click into the batch. Confirm the sample table shows a mix of:
   - `random` rows,
   - `low_confidence` rows where OCR confidence < `hybrid_escalate_below`,
   - `ai_flagged` rows where the LLM gated its own score (rare — only if the
     rubric hard-codes `requires_human_review`),
   - `override` rows = entries where the human already overrode.
4. Approve 3 samples and reject 1. The batch must advance to `completed` when
   the last sample is closed.

**Look right?** the sample fractions roughly match
`eval_review_sample_fraction` (default 0.10), state transitions are
linear, no double-review-sample constraint fires.

---

## 6. Settings page (where feature flags live)

1. Go to `/settings`.
2. Change `reader_mode` from `hybrid` to `tesseract`. Click Save.
3. Run a new entry on `/sandbox`. The reader name in the result must now say
   `tesseract`.
4. Restore `reader_mode = hybrid`. Confirm it persists across reload.

`docs/architecture.md` already documents which flags are live; the Settings
page is the runtime knob.

---

## 7. Where to file evidence

For any non-trivial test you ran, drop evidence under `docs/screenshots/`
using the existing naming convention (`<page>_<state>.png`). Re-run
`make screenshots` to capture the live UI; the Playwright sweep already
exercises the four flows.

If something broke, file:

```
docs/bugs/<date>_<short-slug>.md
```

with: steps to reproduce, expected vs actual, a screenshot, and the
relevant log line from `make logs`.

---

## 8. Honest checklist (what we have NOT measured)

Before you trust the score number on the screen, read `docs/limitations.md`.
The short version:

- No real client data has been benchmarked. All reader/scoring numbers come
  from synthetic pages.
- Marathi is a `TESSERACT_LANGS=eng+hin+mar` switch plus a `mar` traineddata
  file — not shipped by default in the image.
- Single-operator tool. No user accounts. `127.0.0.1` only. Add auth before
  exposing port 8090 to a network (proposal in `docs/roadmap.md` § Auth).
- Vision path needs internet. Offline falls back to tesseract only.

If a feature flag would change any of the above, that change is a settings
toggle + a screenshot, never a code change to the critical path.
