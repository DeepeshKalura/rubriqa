# OCR tools — research and roadmap

This is the post-no-legacy view of what we read with, why, and what we'd
add next. It does NOT contradict `docs/architecture.md` § OCR; it extends
it with options that did not exist (or did not fit) when that section
was written.

The hard constraints come from `TASTE.md` P4 (16 GB office laptop, no
Redis/K8s) and P3 (evidence over claims):

- the reader must run inside the existing `web` image without blowing
  past ~2 GB of new layers,
- the reader must be open-source with an OSI-approved licence,
- before adding it as the default, we need measured numbers on real
  pages (not synthetic), in the same `make bench-readers` harness.

---

## 1. What we read with today

| Reader | How it runs | Marathi? | Cost per page | Where the decision lives |
|---|---|---|---|---|
| `pdf_text` | pure Python, no model | n/a (text-layer only) | ~0 ms | `app/pipeline/readers/pdf_text.py` |
| `tesseract` | CPU, `tessdata` traineddata files | needs `mar.traineddata` | ~2 s | `app/pipeline/readers/tesseract.py` |
| `minimax_vision` | MiniMax M3 hosted (`/v1/chat/completions`) | yes (LLM sees glyphs) | ~13 s + tokens | `app/pipeline/readers/minimax_vision.py` |
| `hybrid` | chain of the above with confidence escalations | inherits from chain | 2–13 s | `app/pipeline/readers/hybrid.py` |

The hybrid chain is the production default. Marathi support is a
`TESSERACT_LANGS=eng+hin+mar` switch + a `mar.traineddata` file
(see `docs/limitations.md` #10).

---

## 2. Candidate open-source readers

| Reader | Licence | Weights / image delta | CPU ok? | Devanagari | Hindi handwriting | Marathi | Last release | Fit for us? |
|---|---|---|---|---|---|---|---|---|
| PaddleOCR `PP-OCRv5_mobile` | Apache-2.0 | ~12 MB on disk | yes | trained | good (printed); limited (handwriting) | yes (`lang=mr` model) | 2025 | **yes** — small, CPU, trained for Marathi |
| EasyOCR | Apache-2.0 | ~250 MB on disk | yes | yes (via `hi`, `mr` packs) | good | yes | rolling | medium — multi-language is its strong suit, but image size keeps it as a `feature_flag` candidate |
| docTR | Apache-2.0 | ~150 MB | yes | limited (Latin + a few) | limited | limited | 2024 | no — weak Devanagari |
| Surya `surya-ocr` | GPL-3.0 | ~1.3 GB | yes (slow) | trained | very good | partial | 2025 | no — image size + GPL vs our Apache-2.0 default |
| Qwen2.5-VL-3B-Instruct | Apache-2.0 | ~6 GB (int4 ~3 GB) | no (GPU) | strong | very good | strong | 2025 | behind a flag only — needs GPU or hosted endpoint |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 | ~14 GB | no (GPU) | strongest | strongest | strong | 2025 | behind a flag only |
| GOT-OCR2-0.5B | Apache-2.0 | ~1 GB | marginal | strong | good | good | 2024 | marginal — the 0.5B is light but slow on CPU |
| OlmOCR-7B | Apache-2.0 | ~14 GB | no (GPU) | good (LR data oriented) | good | good | 2025 | behind a flag only |
| `tesseract` + Marathi traineddata | Apache-2.0 | +12 MB | yes | yes | fair | yes | stable | **add now** — cheapest path to Marathi parity |

PaddleOCR is the only entry that lands inside the existing budget *and*
ships Marathi weights. Qwen2.5-VL is the strongest vision quality, but
it needs a GPU host or a paid inference endpoint — feature-flag it
behind `READER_MODE=qwen_vl_local` or `READER_MODE=qwen_vl_hosted`.

---

## 3. Plan to add Marathi (cheapest first)

1. Add `mar.traineddata` to the tesseract data dir in the image.
   - Verify checksum from <https://github.com/tesseract-ocr/tessdata>.
   - Drop the file into `docker/tessdata/mar.traineddata`; `make
     build` picks it up.
2. Flip `TESSERACT_LANGS=eng+hin+mar` in `.env.example`.
3. Re-run `make bench-readers --data data/samples/marathi/`. Capture:
   - answer recall, CER, p50 wall-clock per page,
   - failure modes (matra dropped, danda misread, anusvara / chandrabindu
     swaps).
4. Pick a small folder of real handwritten Marathi pages from
   `data/samples/pages/` and add it to the bench folder.
5. If recall is below the same Hindi baseline, the next move is PaddleOCR
   mobile as a second tier reader (not a replacement).

This matches ADR-0001: feature-flag the change, don't hardcode it, prove
the number on real pages.

---

## 4. Plan to evaluate one open-weights VLM

The vision path's biggest cost is internet dependency and per-call
latency. A locally runnable open-weights VLM removes both, at the cost of
GPU. The constraint: it must never become the default on a CPU laptop.

1. Stand up a separate compose profile `docker-compose.gpu.yml` that runs
   vLLM (Apache-2.0) with `Qwen2.5-VL-3B-Instruct` (int4) on the GPU
   host. Document in `docs/setup.md` § Local VLM.
2. Add `QwenVlClient` in `app/providers/llm/` with the same interface
   `LLMProvider` exposes today (so swapping `READER_MODE=qwen_vl_hosted`
   does not touch scoring, cleaning, or templates).
3. Wire `READER_MODE=qwen_vl_hosted` into the reader factory in
   `app/pipeline/readers/__init__.py` behind the same confidence
   contract.
4. Benchmark on the same real pages. Compare against MiniMax M3
   baseline. Report answer recall, CER, mean / p95 latency, tokens per
   page, $/page at hosted rates, peak VRAM.
5. Only if quality ≥ current hybrid on Marathi handwriting AND $/page ≤
   current hybrid does it graduate from `feature_flag` to a default
   option. Otherwise it stays a flag.

---

## 5. Testing scripts we need

Each reader must have a deterministic bench and a smoke runnable
without GPU, without internet, on the dev laptop.

```
scripts/bench_readers.py        # current (deleted by ADR-0001 — recreate, smaller)
scripts/bench_readers_smoke.py  # 3-page smoke run, < 30 s, exit code = pass/fail
scripts/ocr_marathi_smoke.py    # asserts Marathi traineddata is loaded and 'mar' joins the lang list
scripts/readers_health.py       # /healthz-style endpoint that reports reader mode + last bench timestamp
```

All four run inside Docker (`make bench-readers-smoke`,
`make bench-marathi-smoke`, `make readers-health`). None depend on a
GPU. The full bench is gated behind `--full`.

---

## 6. What we explicitly reject

- **Surya** in the default image. The torch + model delta is ~1.3 GB and
  it pulls a GPL dependency. Reopen the question only if the budget is
  formally relaxed in `TASTE.md`.
- **OCR-only custom models.** Tesseract's Devanagari model already
  generalises; a from-scratch traineddata file is a 6-month side quest.
- **Cloud-only readers without a local fallback.** That breaks
  `docs/limitations.md` #3 (offline fallback).

---

## 7. Open questions to resolve before adding anything

1. Where do the Marathi `data/samples/marathi/` pages come from? (Own
   collection, partner org, public dataset — needs provenance.)
2. What's the GPU story for this laptop in 2026? If there's no GPU host
   in the loop, Qwen-VL stays a hosted-only flag forever.
3. Does `TASTE.md` P4 relax to "≤ 4 GB image, ≤ 14 GB model loaded in
   RAM" if we gate heavy models behind a `feature_flag`? Currently the
   constraint reads as unconditional.

Once those three are answered, PaddleOCR-mobile and Marathi traineddata
are the natural next two changes; Qwen-VL sits behind them in the
queue.
