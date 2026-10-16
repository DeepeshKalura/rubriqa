"""Benchmark local OCR engines (Tesseract vs PP-OCRv3 ONNX) on the golden set.

This is the Stage 2 measurement for issue #9. It answers one question: does an
alternative local engine beat Tesseract on our 11 human-verified pages?

Why this is a standalone script first, not a pipeline change: the whole point
is to *measure* before we integrate. If PP-OCR cannot beat Tesseract's measured
84.8% mean CER, it must not enter the pipeline at all. Only after it wins do
we add a reader behind the DocumentReader protocol.

PP-OCRv3 (ONNX, Apache-2.0) is used for two reasons: it is a neural engine
(Tesseract is an LSTM trained on printed text and loses on handwriting), and it
installs via onnxruntime (~24 MB) without a multi-GB torch/paddle runtime.

CER is computed with the same routine the sandbox uses
(app/routers/sandbox.py::_char_error_rate) so numbers are directly comparable
to every prior report.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Prefer the live bind mount over the image's baked copy, so edits to
# data/samples take effect without rebuilding. See docker-compose.yml (`./data:/data`).
_SAMPLES = Path("/data/samples") if Path("/data/samples/pages").is_dir() else Path("data/samples")
PAGES_DIR = _SAMPLES / "pages"
GOLDEN_DIR = _SAMPLES / "golden"
MODEL_DIR = Path("/tmp/ppocr")


# ---------------------------------------------------------------------------
# CER — identical method to the sandbox endpoint
# ---------------------------------------------------------------------------
def char_error_rate(reference: str, hypothesis: str) -> float | None:
    import re as _re

    def norm(t: str) -> str:
        return _re.sub(r"\s+", " ", t.lower()).strip()

    ref, hyp = norm(reference), norm(hypothesis)
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        cur = [i]
        for j, hc in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (0 if rc == hc else 1)))
        prev = cur
    return prev[-1] / len(ref)


# ---------------------------------------------------------------------------
# Tesseract (baseline)
# ---------------------------------------------------------------------------
def run_tesseract(image: Path, langs: str, psm: int) -> str:
    out = subprocess.run(
        ["tesseract", str(image), "stdout", "-l", langs, "--psm", str(psm)],
        capture_output=True, text=True, timeout=180,
    )
    return out.stdout


def _runs(idx):
    """Split a sorted index array into contiguous (start, end) runs."""
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


# ---------------------------------------------------------------------------
# PP-OCRv3 ONNX pipeline (det -> crop -> rec -> CTC decode)
# ---------------------------------------------------------------------------
def _connected_row_bands(mask, dilate_frac: float = 0.012, min_h: int = 6):
    """Separate text lines with a vertical dilation + connected components.

    A plain row profile merges handwriting lines that touch. Dilating the mask
    downward by a fraction of the page height bridges the inter-line gaps into
    a single blob per line; counting how many times a given row is covered
    splits those blobs back into one band each.

    Returns a list of (row_start, row_end) in original mask coordinates, or []
    when the mask is empty.
    """
    np = mask.shape and None  # placeholder to keep numpy local; real np below
    import numpy as _np

    m = _np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    H, W = m.shape
    k = max(1, int(H * dilate_frac))
    # dilate downward (and slightly up) via shifted ORs
    d = m.copy()
    for s in range(1, k + 1):
        d[:-s] |= m[s:]
        d[s:] |= m[:-s]
    # count coverage runs: a row is "line interior" if it is covered
    rows_active = d.any(axis=1)
    idx = _np.where(rows_active)[0]
    if idx.size == 0:
        return []
    bands = []
    start = prev = idx[0]
    for r in idx[1:]:
        if r - prev > 1:
            bands.append((start, prev))
            start = r
        prev = r
    bands.append((start, prev))
    return [(a, b) for (a, b) in bands if (b - a) >= min_h]
class PPOCR:
    """Minimal PP-OCRv3 inference using onnxruntime directly."""

    def __init__(self, model_dir: Path, lang: str = "hindi"):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        det = model_dir / "det_v3.onnx"
        rec = model_dir / f"rec_{'deva' if lang == 'hindi' else 'eng'}.onnx"
        dict_file = model_dir / f"dict_{'deva' if lang == 'hindi' else 'eng'}.txt"
        for f in (det, rec, dict_file):
            if not f.exists():
                raise FileNotFoundError(f"missing model file: {f}")
        # CPU only, deterministic, single thread for reproducibility
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.det = ort.InferenceSession(str(det), opts, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(rec), opts, providers=["CPUExecutionProvider"])
        # CTC dict: blank at index 0
        self.charset = ["<blank>"] + [ln.rstrip("\n") for ln in dict_file.read_text(encoding="utf-8").splitlines() if ln != ""]

    def _preprocess_det(self, img):
        """img: HxWx3 uint8 array -> square, normalised NCHW float32.

        PP-OCRv3 detection runs on a SQUARE canvas (the model broadcasts a
        fixed pad over the map, so width must equal height or an Add node
        fails). The source image is letterboxed into that square, and scale_x
        / scale_y are returned so detections can be mapped back.
        """
        np = self.np
        from PIL import Image

        RESAMPLE = Image.Resampling.BILINEAR
        h, w = img.shape[:2]
        side = 960 if max(h, w) > 960 else 640
        scale = side / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        resized = np.array(Image.fromarray(img).resize((nw, nh), RESAMPLE))
        # letterbox into a square canvas
        canvas = np.zeros((side, side, 3), dtype=resized.dtype)
        canvas[:nh, :nw] = resized
        arr = canvas.astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)[None, ...]
        return arr, nw / w, nh / h

    def detect(self, img) -> list[list[float]]:
        np = self.np
        arr, scale_x, scale_y = self._preprocess_det(img)
        out = self.det.run(None, {self.det.get_inputs()[0].name: arr})[0]
        prob = out[0, 0]
        thresh = 0.3
        mask = prob > thresh
        if not mask.any():
            return []

        # --- row profiling -------------------------------------------------
        # Recover one band per text line. A global density threshold merges
        # touching lines; dilation over-merges within a line. So: smooth the
        # row-ink profile and cut at *local minima* — the valleys between
        # lines — which works on both clean and slanted pages.
        density = mask.sum(axis=1).astype("float32")
        if density.max() <= 0:
            return []
        # smooth with a short moving average to wash out intra-line dips
        win = max(3, int(mask.shape[0] * 0.012) | 1)
        kernel = np.ones(win, dtype="float32") / win
        smooth = np.convolve(density, kernel, mode="same")

        # find contiguous runs above a low floor
        floor = max(1.0, float(smooth.max()) * 0.06)
        above = smooth > floor
        idx = np.where(above)[0]
        if idx.size == 0:
            return []

        # split each run at local minima (valleys) deeper than 25% of the
        # neighbouring peaks
        bands: list[tuple[int, int]] = []
        for (a, b) in _runs(idx):
            if b - a < 5:
                continue
            seg = smooth[a:b + 1]
            cuts = [0]
            n = len(seg)
            for i in range(2, n - 2):
                if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1] and seg[i] < seg[max(0, i - 8):i + 8].max() * 0.5:
                    cuts.append(i)
            cuts.append(n - 1)
            for i in range(len(cuts) - 1):
                s, t = a + cuts[i], a + cuts[i + 1]
                if t - s >= 5:
                    bands.append((s, t))
        if not bands:
            bands = _runs(idx)

        h, w = mask.shape
        boxes: list[list[float]] = []
        for (r0, r1) in bands:
            if r1 - r0 < 5:
                continue
            band = mask[r0:r1 + 1]
            cols = np.where(band.any(axis=0))[0]
            if cols.size == 0:
                continue
            # split a line into words only on a wide gap (>= ~4% of width)
            gap = max(24, int(w * 0.04))
            c0, prev_c = cols[0], cols[0]
            segs: list[tuple[int, int]] = []
            for c in cols[1:]:
                if c - prev_c > gap:
                    segs.append((c0, prev_c))
                    c0 = c
                prev_c = c
            segs.append((c0, prev_c))
            for (x0, x1) in segs:
                box = [x0 / scale_x, r0 / scale_y, x1 / scale_x, r1 / scale_y]
                if (box[2] - box[0]) > 12 and (box[3] - box[1]) > 10:
                    boxes.append(box)

        # reading order: top-to-bottom, then left-to-right
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    def recognise(self, crop) -> tuple[str, float]:
        np = self.np
        from PIL import Image

        RESAMPLE = Image.Resampling.BILINEAR
        img = crop.convert("RGB")
        w, h = img.size
        target_h = 48  # PP-OCRv3 recognition height (from the ONNX input shape)
        target_w = max(16, int(w * target_h / h))
        target_w = min(target_w, 320)
        img = img.resize((target_w, target_h), RESAMPLE)
        arr = np.array(img).astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)[None, ...]
        out = self.rec.run(None, {self.rec.get_inputs()[0].name: arr})[0]  # [1, T, C]
        logits = out[0]
        idx = logits.argmax(axis=1)
        conf = logits.max(axis=1)
        # CTC greedy decode
        chars, prev = [], -1
        for t, c in enumerate(idx):
            if c != prev and c != 0:
                if 0 <= c < len(self.charset):
                    chars.append(self.charset[c])
            prev = c
        return "".join(chars), float(conf.mean())

    def read(self, image_path: Path) -> str:
        from PIL import Image

        np = self.np
        img = Image.open(image_path).convert("RGB")
        boxes = self.detect(np.array(img))
        if not boxes:
            return ""

        # Group boxes into lines (rows within ~1.2 line heights share a line),
        # then recognise each line as a single crop. Recognising whole lines
        # instead of isolated words is what lets the recogniser use context.
        boxes.sort(key=lambda b: (b[1], b[0]))
        heights = [b[3] - b[1] for b in boxes if b[3] > b[1]]
        tol = (sum(heights) / len(heights) * 0.6) if heights else 12.0

        lines: list[list[list[float]]] = [[boxes[0]]]
        for b in boxes[1:]:
            prev = lines[-1][-1]
            same_line = abs(b[1] - prev[1]) <= tol
            lines[-1].append(b) if same_line else lines.append([b])

        out_lines: list[str] = []
        for group in lines:
            x0 = int(min(b[0] for b in group))
            y0 = int(min(b[1] for b in group))
            x1 = int(max(b[2] for b in group))
            y1 = int(max(b[3] for b in group))
            # Guard: a single "line" spanning most of the page means detection
            # failed to segment. Recognising it would squash the whole page to
            # 48px and return garbage, so drop it rather than emit noise.
            if (x1 - x0) > img.width * 0.95 and (y1 - y0) > img.height * 0.8:
                continue
            pad = 6
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(img.width, x1 + pad)
            y1 = min(img.height, y1 + pad)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            txt, _ = self.recognise(img.crop((x0, y0, x1, y1)))
            if txt.strip():
                out_lines.append(txt.strip())
        return "\n".join(out_lines)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/ocr_engine_bench.json")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--engines", nargs="*", default=["tesseract", "ppocr_deva", "ppocr_eng"])
    ap.add_argument("--tess-psm", type=int, default=11)
    args = ap.parse_args()

    pages = sorted(PAGES_DIR.glob("page-*.png"))
    if not pages:
        print("no sample pages found", file=sys.stderr)
        return 1

    # engines: name -> ("tess", psm) | ("ppocr", PPOCR instance)
    engines: dict[str, Any] = {}
    if "tesseract" in args.engines:
        engines["tesseract"] = ("tess", args.tess_psm)
    if any(e.startswith("ppocr") for e in args.engines):
        try:
            engines["ppocr_deva"] = ("ppocr", PPOCR(Path(args.model_dir), lang="hindi"))
            engines["ppocr_eng"] = ("ppocr", PPOCR(Path(args.model_dir), lang="english"))
        except Exception as exc:  # noqa: BLE001
            print(f"PP-OCR unavailable: {exc}", file=sys.stderr)
            if "tesseract" not in engines:
                return 1

    rows: list[dict] = []
    for page in pages:
        golden_file = GOLDEN_DIR / f"{page.stem}.txt"
        if not golden_file.exists():
            continue
        golden = golden_file.read_text(encoding="utf-8").strip()
        for name, eng in engines.items():
            kind, payload = eng
            t0 = time.perf_counter()
            try:
                if kind == "tess":
                    text = run_tesseract(page, "eng+hin", payload)
                else:
                    text = payload.read(page)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name} {page.stem} failed: {exc}", file=sys.stderr)
                continue
            elapsed = time.perf_counter() - t0
            cer = char_error_rate(golden, text)
            rows.append({
                "page": page.stem,
                "engine": name,
                "cer": round(100 * cer, 1) if cer is not None else None,
                "elapsed_ms": int(elapsed * 1000),
                "chars": len(text),
                "text": text,
            })
            print(f"  {name:<12} {page.stem}  CER {rows[-1]['cer']}  {elapsed:.1f}s  {rows[-1]['chars']} chars", file=sys.stderr)

    # --- summary ---
    print()
    print("=" * 88)
    print("LOCAL OCR ENGINE BENCHMARK  (lower CER is better)")
    print("=" * 88)
    print(f"{'engine':<14} {'n':>3} {'mean CER':>9} {'median':>8} {'best':>7} {'worst':>7} {'mean ms':>8}")
    summary = []
    for name in engines:
        vals = [r["cer"] for r in rows if r["engine"] == name and r["cer"] is not None]
        times = [r["elapsed_ms"] for r in rows if r["engine"] == name]
        if not vals:
            print(f"{name:<14} {0:>3}  (no output)")
            summary.append({"engine": name, "n": 0})
            continue
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        mean_ms = sum(times) / len(times) if times else 0
        print(f"{name:<14} {len(vals):>3} {mean:>8.1f}% {vals_sorted[len(vals)//2]:>7.1f}% {min(vals):>6.1f}% {max(vals):>6.1f}% {mean_ms:>8.0f}")
        summary.append({
            "engine": name, "n": len(vals),
            "mean_cer": round(mean, 1),
            "median_cer": round(vals_sorted[len(vals)//2], 1),
            "best_cer": round(min(vals), 1),
            "worst_cer": round(max(vals), 1),
            "mean_ms": int(mean_ms),
        })

    # per-page head-to-head
    print()
    print(f"{'page':<10} " + " ".join(f"{n:>13}" for n in engines))
    for page in pages:
        cells = []
        for name in engines:
            r = next((x for x in rows if x["page"] == page.stem and x["engine"] == name), None)
            cells.append(f"{(f'{r['cer']:.1f}' if r and r['cer'] is not None else '-'):>13}")
        print(f"{page.stem:<10} " + " ".join(cells))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows, "tess_psm": args.tess_psm}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
