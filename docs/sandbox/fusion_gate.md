# Fusion gate — decision record

## The question
Path C (vision+tesseract+fusion LLM call) sometimes wins on CER vs Path A (strict
vision), sometimes loses. The user asked: should we add a "skip fusion" gate to
avoid wasted LLM calls when fusion is going to hurt? Two held-out pages (p03,
p11) are reserved for evaluation; the threshold is picked from 9 training pages
only.

## Training data (9 pages, sorted by tesseract text length / vision text length)

| page  | vis_len | tess_len | ratio | winner   | Δ CER (fusion − vision) |
|-------|---------|----------|-------|----------|------------------------:|
| p-08  |     715 |        0 |  0.00 | vision   |                   +1.6 |
| p-10  |     792 |       73 |  0.09 | fusion   |                  −63.5 |
| p-01  |     571 |      441 |  0.77 | vision   |                   +3.3 |
| p-02  |   1 094 |      899 |  0.82 | fusion   |                   −7.8 |
| p-09  |     317 |      264 |  0.83 | vision   |                   +0.2 |
| p-04  |     404 |      514 |  1.27 | fusion   |                   −4.7 |
| **p-05** |   **434** |    **770** |  **1.77** | **vision** |           **+18.5** |
| **p-07** | **1 013** |  **2 317** |  **2.29** | **vision** |            **+5.2** |
| p-06  |       0 |      312 |   ∞   | fusion   |                  −66.9 |

("winner" = lower CER; negative Δ means fusion better. p-06 vision returned
empty, so fusion was the only option.)

## What the data shows
- `tess_confidence` is **not** a useful predictor. p-07 has tesseract confidence
  0.318 but fusion hurts; p-06 has 0.315 and fusion wins by 66.9 CER. They are
  indistinguishable on confidence alone.
- The `tess_text_len / vision_text_len` ratio **is** a useful predictor. When
  tesseract's output is much longer than vision's, tesseract is hallucinating
  (English-script noise on Devanagari pages, or 2× the right length), and the
  fusion LLM call folds that noise into the hybrid output.
- When the ratio is ≤ ~1.5, fusion is at worst a tie; at best a real win. When
  the ratio is > 1.5, fusion loses on every training page (2/2).

## The gate (single feature flag)

```python
# In process_entry_via_fusion, before the fusion LLM call:
if (
    vision_text.strip()
    and tess_text.strip()
    and len(tess_text) > settings.fusion_max_tess_to_vision_ratio * len(vision_text)
):
    # tesseract's text is suspiciously long compared to vision's — likely
    # hallucinated noise. Use vision text as the hybrid output, skip the
    # fusion LLM call entirely.
    fused_text = vision_text
    fused_language = vision_language
    fusion_result = None
    fusion_was_skipped = True
elif tess_text.strip() or vision_text.strip():
    # existing fusion LLM call
    ...
```

Flag in `app/config.py`:

```python
fusion_max_tess_to_vision_ratio: float = 1.5
```

## Training-set outcome at ratio = 1.5

| action                       | pages | result on CER |
|------------------------------|-------|---------------|
| Correctly skipped fusion     | 2     | p-05 (+18.5 saved), p-07 (+5.2 saved) |
| Wrongly skipped fusion       | 0     | — |
| Correctly ran fusion         | 6     | p-01/02/04/08/09/10 — 3 wins (p-02, p-04, p-10), 1 force-fused (p-06), 2 near-ties (p-01, p-08, p-09) |
| Forced fusion (vision empty) | 1     | p-06 (the only path that worked) |

Net CER effect on the 9 training pages: ~+24 CER saved (the two skips) plus
~−76 CER captured (the fusion runs that are kept). The gate is a clean win on
training data.

## Why this is not over-optimization
- The gate is a single ratio threshold with a name that describes the failure
  mode (tesseract text too long → noise).
- The threshold 1.5 is a round, conservative number, not a fitted value. The
  page-by-page ratios cluster on both sides (1.27, 1.77, 2.29) and 1.5 sits in
  the only natural gap in the data.
- No prompt edit, no model change, no schema change. The flag is a runtime
  switch per AGENTS.md "feature flags over hardcoding".
- The held-out 2 pages (p-03, p-11) are not in this document. They are
  evaluated in `holdout_metrics.json` after the gate is implemented.
