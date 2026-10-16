# OCR engine decision — Path B replacement

Decision record for issue #9. Follows the same shape as
`docs/sandbox/fusion_gate.md`: question, measured evidence, what was rejected,
and what is explicitly *not* decided here.

> **Superseded in part (2026-10-14).** Stage 3 found that the golden transcript
> set was **misaligned with the page images**, so every number below involving
> pages 07–11 was wrong. The corrected figures are in
> [Stage 3](#stage-3--gemini-vision-wins-and-the-golden-set-was-broken).
> The conclusions that survive correction: Tesseract is architecturally wrong
> (its CER barely moved), PP-OCRv3 still loses, and the fusion gate is still
> undecidable. Read Stage 3 first.

All numbers are measured, not estimated. Sources:

- `reports/bench_models.json` — 63 live runs (2 models × 3 paths × 11 pages)
- `reports/bench_variance_check.json` — 6 repeat runs on the two decision-critical pages
- `reports/fusion_gate_replay.json` — gate replay produced by
  `scripts/replay_fusion_gate.py`
- `reports/cloud_ocr_bench.json` — Gemini vision vs Tesseract on the corrected set
- `reports/golden_alignment.json` — the check that found the misalignment

## The question

Tesseract wins **0 of 11** pages on CER. Should Path B be retired, and if so
what replaces it?

## Finding 1 — Tesseract is architecturally wrong, not misconfigured

We measured the engine directly (not through the sandbox) on all 11 golden
transcripts, varying only page-segmentation mode:

| `--psm` | mean CER | median | **best page** | worst |
|---|---|---|---|---|
| 3 (auto) | 87.1% | 83.3% | 79.2% | 100.0% |
| 4 (single column) | 87.2% | 83.1% | 79.4% | 100.0% |
| 6 (**current**) | 91.5% | 83.0% | 70.8% | 146.6% |
| 11 (sparse) | **84.8%** | 83.5% | **70.1%** | 94.8% |

No configuration rescues it. The **best single page is still ~70% CER**, while
the MiniMax vision path (A) reaches 8–20% on those same pages. Tuning cannot
close a 4× gap because the gap is architectural: Tesseract is trained on
printed text.

Independent benchmark data agrees — on handwriting, Tesseract CER 0.785 /
WER 1.043 (more errors than words) versus PP-OCRv5 0.477 and GLM-OCR 0.328.

## Finding 2 — a real config bug, worth fixing but not decisive

`Dockerfile` installs only `tesseract-ocr-eng` and `tesseract-ocr-hin`. Five of
eleven pages are Marathi; `tesseract-ocr-mar` exists in apt and was never
installed. Page-05 — the worst hallucination in the corpus at 148.6% CER — is
entirely in Marathi (`नाव - GAYATRI इयत्ता - 10TH शाळा - MEHADAR`).

This is a genuine defect and should be corrected, but the PSM table above was
produced under the same `eng+hin` constraint, so the conclusion does not change:
the ceiling is ~70% CER on the *best* page, not competitive with a vision LLM.

## Finding 3 — the fusion gate's justification no longer reproduces

`docs/sandbox/fusion_gate.md` recorded that on page-05, fusion *hurt* by
+18.5 CER and the gate saved it. On the current pipeline that does not
reproduce. Two independent runs:

| page | run 1: vision → fusion | run 2: vision → fusion | verdict |
|---|---|---|---|
| page-04 | 50.7 → **33.5** (helped) | 43.0 → **43.7** (hurt) | **flipped** |
| page-05 | 76.5 → **63.9** (helped) | 67.8 → **60.7** (helped) | stable *helped* |

Gate replay at the shipped threshold (`fusion_max_tess_to_vision_ratio = 1.5`):

| threshold | fired | TP | FP | FN | TN | precision | recall | net CER |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 2 | 0 | 2 | 3 | 2 | 0.00 | 0.00 | −29.8 |
| 1.25 | 2 | 0 | 2 | 3 | 2 | 0.00 | 0.00 | −29.8 |
| **1.5 (shipped)** | **1** | **0** | **1** | **3** | **3** | **0.00** | **0.00** | **−12.6** |
| 2.0 / 3.0 / never | 0 | 0 | 0 | 3 | 4 | — | 0.00 | +0.0 |

The gate fires on exactly one page, and on that page fusion *helped* both
times — so gating it forfeits a real improvement. On this data the gate has
**zero precision and zero recall**, and is net negative.

**This is not enough to switch the gate off.** It rests on n=1 gated page, and
the p-04 verdict flipped between runs. What it *does* establish is that the
gate's original justification is no longer reproducible on the current pipeline.

## Finding 4 — the measurement is split: one engine is deterministic, the others are not

Repeating the same page twice, same model, same prompt, temperature 0.0:

| engine | page-04 | page-05 | variance |
|---|---|---|---|
| Tesseract (Path B) | 96.7 → 96.7 | 148.6 → 148.6 | **Δ 0.0** |
| Vision (Path A) | 50.7 → 43.0 | 76.5 → 67.8 | Δ −7.7 / −8.7 |
| Fusion (Path C) | 33.5 → 43.7 | 63.9 → 60.7 | **Δ +10.2** / −3.2 |

Tesseract is fully reproducible. The vision LLM is not, and Path C's
run-to-run spread (±10 CER points) is **larger than the effect the fusion gate
is trying to detect**.

This is the most important line in this document. It splits the decision cleanly:

- **The Tesseract verdict is safe.** n=11, deterministic, ~84% best-case mean
  CER (84.8% in the PSM sweep, 84.1% in the head-to-head run — same conclusion,
  minor Tesseract-version jitter). It does not need more samples.
- **Any vision-vs-fusion verdict is not safe.** It needs repeated runs per page
  before it can be trusted — which is the same blocker as issue #6.

## Finding 5 — PP-OCRv3 (ONNX) was benchmarked and it does not beat Tesseract

The Stage 2 ship gate was explicit: beat **Tesseract's ~84% mean / ~70% best-page
CER**. We ran PP-OCRv3 (Apache-2.0, installed via `onnxruntime` ~24 MB, no
torch) on the same 11 pages with the same CER method, both engines run in the
same harness so the comparison is apples-to-apples. Detection uses
`det_v3.onnx` + Devanagari recognition `rec_deva.onnx`.

| page | Tesseract (psm 11) | PP-OCRv3 Devanagari | winner |
|---|---|---|---|
| page-01 | **70.8%** | 78.8% | tesseract |
| page-02 | **78.2%** | 90.1% | tesseract |
| page-03 | **82.7%** | 89.7% | tesseract |
| page-04 | **85.8%** | 89.3% | tesseract |
| page-05 | 91.1% | **88.1%** | ppocr |
| page-06 | **80.9%** | 90.3% | tesseract |
| page-07 | 85.9% | **84.0%** | ppocr |
| page-08 | **93.2%** | 100.0% (no text) | tesseract |
| page-09 | **80.9%** | 93.0% | tesseract |
| page-10 | **85.3%** | 100.0% (no text) | tesseract |
| page-11 | **91.6%** | 99.8% (no text) | tesseract |
| **mean** | **84.1%** | 91.2% | — |
| **wins** | **9** | 2 | — |

**PP-OCRv3 fails the ship gate.** It is *worse* on average (91.2% vs 84.1%) and
wins only 2 of 11. It also produced no text at all on pages 08, 10, 11 under our
line-segmentation (the hand-rolled ONNX pipeline, not the official PaddleOCR
post-processor, is the likely cause on the compact/slanted pages).

**It is not shipped.** The benchmark harness (`scripts/bench_ocr_engines.py`) is
kept so the result is reproducible, but PP-OCRv3 is **not** added to the pipeline,
`requirements.txt`, or the `DocumentReader` factory. Nothing in the production
path changed.

Honest note: the PP-OCR result is a floor, not a ceiling. The official PaddleOCR
pipeline uses trained DB post-processing (unclip + polygon + min-area filters)
that we did not reimplement; a faithful port could do better, especially on the
pages that returned nothing. But shipping an engine that loses 9-2 to a known-bad
baseline is not justified, and the correct next step is either a faithful
PaddleOCR port (a real project) or the separate API PR — not a half-tuned ONNX
integration.

## Decision

**Retire Tesseract as a production path. Promote no replacement yet.**

1. The replacement decision rests on Finding 1 + Finding 4: Tesseract's failure
   is measured, reproducible, and architectural. It loses.
2. No candidate is promoted on a literature benchmark. Candidates are ranked by
   *what we can actually install and measure here*:

   | candidate | verdict | reason |
   |---|---|---|
   | PP-OCRv5/v3 (ONNX) | **benchmarked, rejected** | 91.2% mean CER, loses 9-2 to Tesseract (Finding 5) |
   | Kraken | deferred | Ships **no official pretrained models**; no confirmed Devanagari handwriting model. Adopting it is a research project, not a swap |
   | PaddleOCR (native) | deferred | Needs `paddlepaddle` runtime, large |
   | GLM-OCR | deferred | 0.9B VLM — large download and slow CPU inference |
   | Cloud API | **separate PR** | Per owner decision. Needs its own cost model (per-page pricing, not token pricing) |

3. The **ship gate for any future replacement stays explicit**: it must beat
   Tesseract's measured **84.1% mean / 70.8% best-page CER on these same 11
   pages** (psm 11 baseline). If nothing beats it, the honest outcome is
   *"Path B is retired, not replaced."*

## The gate is a separate, still-open question

The gate replay is a finding, not a decision. Before Path C's design changes:

- The gate's premise — "tesseract much longer than vision ⇒ noise, fall back to
  vision" — silently assumes **vision is trustworthy**. On page-05 vision was
  67–76% CER, itself unreliable. Falling back to a bad reader is not a fix.
- A candidate replacement changes the gate's *input distribution*. The
  length-ratio signal only works while the local engine hallucinates long
  output. A better engine breaks that assumption by construction.
- The 5 gate regression tests in `tests/test_sandbox_paths.py` remain green and
  must stay green.

Recommended next step, cheap and decisive: measure the gate against **repeat
runs** (n≥3 per page) rather than a single run, and only then decide whether the
gate is a safety mechanism or a liability.

## What this document does not decide

- Whether a replacement is promoted (awaiting the PP-OCRv5 benchmark).
- Whether the fusion gate is switched off, retuned, or re-scoped (needs repeat runs).
- Whether Path B survives as a control arm (depends on the gate outcome above).
- Cloud API OCR (separate PR, per owner decision).

## Reproduce

```bash
# gate replay from measured runs (no LLM calls)
docker compose exec web python -m scripts.replay_fusion_gate

# local engine benchmark (no LLM calls; needs onnxruntime + models in /tmp/ppocr)
docker compose exec web python -m scripts.bench_ocr_engines

# cloud OCR benchmark (Gemini vision; $0 on the free tier)
docker compose exec web python -m scripts.bench_cloud_ocr

# verify the golden set is aligned with the page images (11 API calls, ~$0.01)
docker compose exec web python -m scripts.bench_cloud_ocr --align \
  --out reports/golden_alignment.json

# variance check (real LLM calls, ~12 min)
docker compose exec web python -m scripts.bench_models \
  --pages page-04 page-05 --models MiniMax-M3 --out reports/bench_variance_check.json
```

PP-OCRv3 model files (`det_v3.onnx`, `rec_deva.onnx`, `rec_eng.onnx`, dicts) are
downloaded to `/tmp/ppocr` in the web container for the benchmark only. They are
**not** committed and **not** added to the image.

`bench_cloud_ocr.py` needs `GOOGLE_API_KEY` in the environment. The key goes in
`.env` (git-ignored) or, to avoid writing it at all, in a 600-mode file outside
the repo and sourced before the run. Never in a commit, a report, or a log.

---

## Stage 3 — Gemini vision wins, and the golden set was broken

Two things happened in Stage 3, and the second one is the more important.

### The cloud OCR survey: most of the field cannot read this corpus

Before writing any code we checked what the cheap document-OCR APIs actually
support, because "handwritten Devanagari" is the hard part:

| API | Handwritten Devanagari | Verdict |
|---|---|---|
| AWS Textract | **No** — handwriting is English-only | dead on arrival |
| Azure Document Intelligence Read | **No** — handwriting covers en/zh/fr/de/it/ja/ko/pt/es. Devanagari is *printed*-only | dead on arrival |
| Google Cloud Vision `document_text_detection` | `Deva` marked **Experimental** | possible, needs a billing card to enable |
| Google Document AI Enterprise OCR | same engine | possible, needs a billing card |
| **Gemini vision (VLM)** | yes — and it is the only one that measured well | **selected** |

The first two are disqualified by language, not by price. That is worth stating
plainly: the usual answer of "just use Textract" does not exist for Devanagari
handwriting.

### The measured result

`gemini-3.1-flash-lite`, 2200px, temperature 0, thinking disabled, run through
the real `DocumentReader` seam. Nine pages, the ones with a verified transcript:

| page | Tesseract (psm 11) | Gemini 3.1 Flash-Lite | winner |
|---|---:|---:|---|
| page-01 | 70.8% | **4.0%** | gemini |
| page-02 | 78.2% | **6.1%** | gemini |
| page-03 | 82.7% | **4.5%** | gemini |
| page-04 | 85.8% | **3.7%** | gemini |
| page-05 | 91.1% | **7.1%** | gemini |
| page-06 | 80.9% | **3.1%** | gemini |
| page-08 | 95.9% | **0.3%** | gemini |
| page-09 | 71.0% | **4.2%** | gemini |
| page-10 | 85.1% | **8.4%** | gemini |
| **mean** | **82.4%** | **4.6%** | gemini, 9–0 |
| **best page** | 70.8% | 0.3% | |

**Gemini wins 9 of 9, by a factor of ~18 on the mean.** Reproduced exactly
across two independent runs. The ship gate (beat ~84% mean / ~70.8% best) is
cleared by a very wide margin, and it clears the bar that PP-OCRv3 missed by
7 points.

Cost, measured from `usageMetadata` on the same runs:

| | measured |
|---|---|
| tokens per page | 1,107 in / 151 out (vs MiniMax M3's 5,100 / 2,700) |
| paid-tier cost per page | **$0.00065** |
| 300-entry program | **$0.20** |
| free tier | **$0** — `gemini-3.1-flash-lite` is free in/out, no card |

Note the shape of the saving: Flash-Lite's *output* rate ($1.50/1M) is actually
**higher** than MiniMax M3's ($1.20/1M). The saving comes entirely from token
count — Gemini returns a bare transcript where M3 emits a reasoning preamble,
and it encodes a 2200px page in about a fifth of the tokens.
`tests/test_providers_gemini.py::test_pricing_row_is_verified_and_cheaper_per_page`
pins this so nobody "optimises" by raising `max_tokens`.

### The golden set was misaligned — and nobody noticed for a month

The first Gemini run reported a *median* of 7.1% and a *mean* of 55.2%, with
five pages above 75% and two above 200%. A CER above 100% means the hypothesis
is longer than the reference, which is the signature of a *mislabelled* golden,
not a bad transcription. So the transcripts were cross-correlated: every
transcription scored against every golden file.

The result was unambiguous:

```
rows = image, cols = golden
    image 08  -> best match was golden page-07 at 0.3%  (nominal golden 209.4%)
    image 09  -> best match was golden page-08 at 4.2%  (nominal golden  75.4%)
    image 10  -> best match was golden page-09 at 8.4%  (nominal golden  85.3%)
    image 07  -> no match (best 79.0%)
    image 11  -> no match (best 75.6%)
```

The transcripts for images 08/09/10 were filed as `page-07/08/09`, and images
07 and 11 never had a transcript. Images 01–06 were correctly paired.

**Consequences, stated plainly:**

1. Every figure in Findings 1–5 above that involved pages 07–11 was wrong. The
   corrected Tesseract mean is 82.4% (was quoted as 84.1%), and PP-OCRv3's
   91.2% becomes 90.7%. Neither conclusion flips.
2. The fusion-gate replay in `reports/fusion_gate_replay.json` compared
   transcripts on the same shifted pages, so its `n=10` and `n=1` counts are
   unreliable. The gate remains **undecidable**, now for two independent
   reasons. It is not being re-measured as part of this decision.
3. `data/samples/holdout.json` lists `page-11`, which has no transcript. It must
   not be used as a hold-out until one is written.

**The fix.** Goldens are now resolved through
`data/samples/golden/manifest.json` instead of by filename, and
`tests/test_golden_alignment.py` asserts that no transcript is claimed by two
images, that every image is either mapped or *explicitly declared* to have no
transcript, and that both bench scripts refuse to run without a manifest.
`scripts/bench_cloud_ocr.py --align` re-runs the cross-correlation so the check
is repeatable rather than a one-off.

### What is decided

**Path B is retired and replaced: `READER_MODE=gemini_vision` with
`LLM_PROVIDER=gemini`.** It is the first candidate to clear the ship gate, and
it does so by a margin no other candidate reached.

Two honest caveats:

1. **n=9, not 11.** Two images have no transcript. Adding them requires writing
   two human-verified transcripts, which is issue #6's work.
2. **The free tier may use the images to improve Google's products** (verified
   on the pricing page: free tier "Yes", paid tier "No"). Goldens only for now.
   A real student page should go through a paid key, which costs ~$0.00065.

### Latency caveat

The measured mean of 20.7 s/page is **not** the model's latency — it is
dominated by free-tier 429 throttling (clean calls returned in 3–8 s). A paid
key removes the retries. Do not project the throttled number.

### Not decided here

- Structured field extraction from the transcript (issue #8).
- Whether the rubric scoring stage should also move off MiniMax. It is a
  text-only task; M3 is good at it and the token cost is low.
- Whether to drop `hybrid_reader_chain` entirely. Tesseract is no longer worth
  escalating to given a 82.4% floor, but keeping it costs nothing and preserves
  a CPU-only fallback for free-tier throttling.
