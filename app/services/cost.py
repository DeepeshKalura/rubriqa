"""Cost, latency and storage projections for OCR path runs.

Prices are USD per **million** tokens, matching how every provider publishes
them. They live in `PRICING` (a feature-flag-friendly table, not code) so a
new model is one row, not a change to the maths.

Every projection here is derived from **measured** runs — see
`scripts/bench_models.py` for the harness and `scripts/bench_cloud_ocr.py` for
the Gemini runs. The rendered report they emit is written by
`scripts/render_bench_report.py` and is **not committed**: the previous
committed copy was built against the retired Tesseract path and the misaligned
golden set, so shipping it would have meant committing numbers known to be
wrong. Re-render it locally when you need it. Two rules this module enforces:

1. Latency projections use a caller-supplied percentile (default p95), never
   the mean. Path C's mean is 41 s and its max is 72 s: a mean-based 25k
   projection is optimistic by ~3x, which is the kind of quiet lie that only
   shows up at scale.
2. Cached input tokens are billed separately from fresh input. MiniMax reports
   prompt caching per call (`usage.prompt_tokens_details.cached_tokens`) and it
   dominates our rubric-scoring calls, where the system prompt is identical on
   every page. Ignoring it would overstate cost several-fold.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --- pricing table (USD per 1M tokens) ---------------------------------------
# `cached_input` falls back to `input` when a provider has no cache tier.
PRICING: dict[str, dict[str, float]] = {
    "minimax-m3": {
        "input": 0.30,
        "cached_input": 0.06,
        "output": 1.20,
        "note": "list price, input <= 512k context",
    },
    "minimax-m3.1-flash-preview": {
        # Preview model. No published list rate at time of writing; these are
        # the M3 rates carried over as a *placeholder* and are flagged
        # unverified so a report can never present them as fact.
        "input": 0.30,
        "cached_input": 0.06,
        "output": 1.20,
        "verified": False,
        "note": "PREVIEW - price unverified, carried over from M3 as placeholder",
    },
    "gemini-3.1-flash-lite": {
        # Executable provider (issue #9 Stage 3). Verified against
        # https://ai.google.dev/gemini-api/docs/pricing on 2026-10-14.
        # Free tier costs $0 and needs no card, but is marked
        # "used to improve our products: Yes"; the paid tier is "No".
        "input": 0.25,
        "cached_input": 0.025,
        "output": 1.50,
        "verified": True,
        "note": "paid standard tier; free tier is $0; batch tier is 50% of this",
    },
    "gemini-3.8-flash": {
        # Report-only reference rate, verified 2026-10-14.
        "input": 0.75,
        "cached_input": 0.075,
        "output": 3.75,
        "verified": True,
        "note": "reference only, not wired up",
    },
}

# Report-only reference rates (no keys configured, never executed). Included so
# the report can show what a provider swap would cost without wiring it up.
# Report-only reference rates (no keys configured, never executed). Included so
# the report can show what a provider swap would cost without wiring it up.
# NOTE: Gemini used to sit here as an unexecuted guess. It is now an executable
# provider with verified rates in PRICING above, so no placeholder is left here
# to be mistaken for fact.
REFERENCE_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku": {"input": 1.00, "cached_input": 0.10, "output": 5.00},
}

# Average on-disk size of one page image, measured from data/samples/pages/
# (11 PNGs, 23.8 MB total).
BYTES_PER_PAGE_IMAGE = int(23_840_857 / 11)


def normalise_model(model: str | None) -> str:
    """Map an API model id onto a PRICING key."""
    if not model:
        return ""
    m = model.strip().lower()
    # Gemini ids arrive bare, prefixed ("models/gemini-..."), or dated
    # ("gemini-2.5-flash-lite-20260101" / "-latest"). Strip only a dated or
    # rolling suffix so we do not mangle real model numbers like "-002".
    m = m.removeprefix("models/").split("@")[0]
    parts = m.rsplit("-", 1)
    if len(parts) == 2 and (parts[1] == "latest" or parts[1].isdigit() and len(parts[1]) == 8):
        m = parts[0]
    if "flash-preview" in m or "m3.1" in m:
        return "minimax-m3.1-flash-preview"
    if m.endswith("m3") or "minimax-m3" in m:
        return "minimax-m3"
    return m


@dataclass
class CostBreakdown:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    @property
    def fresh_input_tokens(self) -> int:
        """Prompt tokens not served from cache (billed at full rate)."""
        return max(0, self.input_tokens - self.cached_tokens)

    def usd(self, model: str | None) -> float:
        price = PRICING.get(normalise_model(model))
        if price is None:
            return 0.0
        return (
            self.fresh_input_tokens / 1_000_000 * price["input"]
            + self.cached_tokens / 1_000_000 * price.get("cached_input", price["input"])
            + self.output_tokens / 1_000_000 * price["output"]
        )

    def as_dict(self, model: str | None) -> dict:
        key = normalise_model(model)
        price = PRICING.get(key, {})
        return {
            "model": model,
            "pricing_key": key,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "fresh_input_tokens": self.fresh_input_tokens,
            "output_tokens": self.output_tokens,
            "cached_share_pct": (
                round(100 * self.cached_tokens / self.input_tokens, 1)
                if self.input_tokens else 0.0
            ),
            "usd": round(self.usd(model), 6),
            "price_verified": price.get("verified", True),
            "price_note": price.get("note", ""),
        }


@dataclass
class RunSample:
    """One measured path run: token counts, latency, CER."""
    path: str
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    cer: float | None = None
    score: float | None = None
    ok: bool = True

    @property
    def cost(self) -> CostBreakdown:
        return CostBreakdown(
            input_tokens=self.input_tokens,
            cached_tokens=self.cached_tokens,
            output_tokens=self.output_tokens,
        )


@dataclass
class PathStats:
    """Aggregate of many measured runs of the same (path, model) pair."""
    path: str
    model: str
    samples: list[RunSample] = field(default_factory=list)

    def add(self, sample: RunSample) -> None:
        self.samples.append(sample)

    # -- latency -------------------------------------------------------------
    def percentile_ms(self, pct: float = 95.0) -> int:
        if not self.samples:
            return 0
        ordered = sorted(s.elapsed_ms for s in self.samples)
        # nearest-rank: index = ceil(pct/100 * n) - 1
        idx = max(0, min(len(ordered) - 1, int(round(pct / 100 * len(ordered))) - 1))
        return ordered[idx]

    def mean_ms(self) -> int:
        return int(statistics.fmean(s.elapsed_ms for s in self.samples)) if self.samples else 0

    # -- cost ----------------------------------------------------------------
    def mean_cost(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.fmean(s.cost.usd(self.model) for s in self.samples)

    def mean_tokens(self) -> tuple[int, int, int]:
        if not self.samples:
            return (0, 0, 0)
        n = len(self.samples)
        return (
            int(statistics.fmean(s.input_tokens for s in self.samples)),
            int(statistics.fmean(s.cached_tokens for s in self.samples)),
            int(statistics.fmean(s.output_tokens for s in self.samples)),
        )

    # -- accuracy ------------------------------------------------------------
    def cer_values(self) -> list[float]:
        return [s.cer for s in self.samples if s.cer is not None]

    def mean_cer(self) -> float | None:
        vals = self.cer_values()
        return statistics.fmean(vals) if vals else None

    def cer_wins(self, winners: Iterable[str]) -> int:
        return sum(1 for s in self.samples if s.path in set(winners))

    def usable_ocr(self) -> int:
        """Runs that actually produced text (a scored run on empty OCR is not a
        working OCR engine — it scored off prefilled form fields)."""
        return sum(1 for s in self.samples if s.ok and s.score is not None)


def project_records(
    stats: PathStats,
    *,
    records: int,
    workers: int = 1,
    latency_percentile: float = 95.0,
) -> dict:
    """Project wall-clock and $ for `records` entries processed by `workers`.

    Uses p95 latency per record divided across workers. Assumes workers are
    fully IO-bound; path B is CPU-bound, so treat its worker-scaling as an upper
    bound (see issue #7).
    """
    n = max(1, stats.usable_ocr() or len(stats.samples))
    per_record_ms = stats.percentile_ms(latency_percentile) / n
    per_record_cost = stats.mean_cost()

    total_serial_ms = per_record_ms * records
    total_wall_ms = total_serial_ms / max(1, workers)

    return {
        "path": stats.path,
        "model": stats.model,
        "records": records,
        "workers": workers,
        "samples": n,
        "per_record_ms_p95": round(per_record_ms),
        "per_record_cost_usd": round(per_record_cost, 6),
        "total_cost_usd": round(per_record_cost * records, 2),
        "serial_hours": round(total_serial_ms / 3_600_000, 2),
        "wall_hours": round(total_wall_ms / 3_600_000, 2),
        "wall_days": round(total_wall_ms / 86_400_000, 2),
        "latency_percentile": latency_percentile,
        # workers required to hit an 8-hour target, for the scale-up decision
        "workers_8h": round(
            (per_record_ms * records) / (8.0 * 3_600_000), 1
        ),
    }


def estimate_run_cost(
    records: int,
    *,
    per_record_cost_usd: float,
    llm_calls_per_record: int = 1,
) -> dict:
    """Pre-run $ estimate for an ingestion run.

    Shown before a run starts so the bill is never a surprise. `records` is the
    batch size, `per_record_cost_usd` a measured figure from a benchmark (not a
    guess), and `llm_calls_per_record` counts billable calls per entry — a
    vision path is 1 read + 1 score, so a path that looks cheap per page can
    double through the scoring stage.
    """
    calls = max(1, records * max(1, llm_calls_per_record))
    total = per_record_cost_usd * records * max(1, llm_calls_per_record)
    return {
        "records": records,
        "llm_calls": calls,
        "per_record_cost_usd": round(per_record_cost_usd, 6),
        "projected_cost_usd": round(total, 4),
        "projected_cost_formatted": fmt_usd(total),
        "confidence": "measured per-record cost from reports/",
    }


class SpendCapExceeded(RuntimeError):
    """Raised when a run would push cumulative spend past the configured cap.

    The cap exists because the cheap-looking failure is the expensive one: an
    entry-flow run that silently loops, retries, or re-processes pages bills
    without a UI saying so. Raise before the call, not after.
    """

    def __init__(self, spent: float, cap: float, stage: str):
        self.spent = spent
        self.cap = cap
        super().__init__(
            f"Spend cap {fmt_usd(cap)} reached ({fmt_usd(spent)} spent) "
            f"before stage '{stage}'. Raise LLM_SPEND_CAP_USD or reduce the "
            f"batch size. Refusing to make a billable call."
        )


class SpendGuard:
    """Cumulative LLM spend for one run, checked before each billable call."""

    def __init__(self, cap_usd: float = 0.0):
        self.cap_usd = cap_usd or 0.0
        self.spent_usd = 0.0
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return self.cap_usd > 0

    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def check(self, stage: str = "llm") -> None:
        """Call before a billable request. No-op when the cap is disabled."""
        if self.enabled and self.spent_usd >= self.cap_usd:
            raise SpendCapExceeded(self.spent_usd, self.cap_usd, stage)

    def record(self, breakdown: CostBreakdown, model: str | None) -> float:
        cost = breakdown.usd(model)
        self.spent_usd += cost
        self.calls += 1
        return cost


def project_storage(records: int, *, bytes_per_page: int = BYTES_PER_PAGE_IMAGE) -> dict:
    """On-disk requirement for `records` page images (originals only)."""
    total = bytes_per_page * records
    return {
        "records": records,
        "bytes_per_page": bytes_per_page,
        "mb_per_page": round(bytes_per_page / 1_048_576, 2),
        "total_gb": round(total / 1_073_741_824, 1),
        "note": "originals only; rendered/derived copies and the golden set add more",
    }


def speedup_needed(
    projection: dict, *, target_hours: float = 8.0, workers_available: int = 8
) -> dict:
    """How many workers a target needs, and whether the current ones suffice."""
    per_record_ms = projection["per_record_ms_p95"]
    records = projection["records"]
    needed = (per_record_ms * records) / (target_hours * 3_600_000)
    return {
        "target_hours": target_hours,
        "workers_needed": round(needed, 1),
        "workers_available": workers_available,
        "feasible": needed <= workers_available,
        "verdict": (
            f"{projection['path']} fits {records:,} records in {target_hours:.0f}h with "
            f"{needed:.1f} workers"
            if needed <= workers_available
            else f"{projection['path']} needs {needed:.1f} workers for {records:,} in "
                 f"{target_hours:.0f}h - only {workers_available} assumed available"
        ),
    }


def fmt_usd(v: float) -> str:
    if v >= 100:
        return f"${v:,.0f}"
    if v >= 1:
        return f"${v:,.2f}"
    if v >= 0.01:
        return f"${v:,.3f}"
    return f"${v:.5f}"


def fmt_gb(v: float) -> str:
    return f"{v:.1f} GB"