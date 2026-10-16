"""Tests for the cost / latency / storage projection engine.

The engine feeds real money decisions (a 25k-record run), so the arithmetic is
pinned here: cached-input pricing must actually be cheaper, p95 must never be
below the mean, and an unknown model must never silently cost $0 in a way that
looks authoritative.
"""
from __future__ import annotations

import pytest

from app.services.cost import (
    BYTES_PER_PAGE_IMAGE,
    CostBreakdown,
    PathStats,
    RunSample,
    normalise_model,
    project_records,
    project_storage,
    speedup_needed,
)


def _stats(path: str, model: str, latencies: list[int], **kw) -> PathStats:
    st = PathStats(path=path, model=model)
    for ms in latencies:
        st.add(
            RunSample(
                path=path,
                model=model,
                input_tokens=kw.get("inp", 1000),
                cached_tokens=kw.get("cached", 0),
                output_tokens=kw.get("out", 1000),
                elapsed_ms=ms,
                cer=kw.get("cer", 10.0),
                score=kw.get("score", 20.0),
            )
        )
    return st


# --- model key normalisation -------------------------------------------------
def test_normalise_model_maps_ids_to_pricing_keys():
    assert normalise_model("MiniMax-M3") == "minimax-m3"
    assert normalise_model("minimax-M3") == "minimax-m3"
    assert normalise_model("MiniMax-M3.1-Flash-Preview") == "minimax-m3.1-flash-preview"
    assert normalise_model("MiniMax-M3.1-Flash-Preview ").__contains__("flash-preview")


def test_normalise_model_unknown_returns_lowercased():
    assert normalise_model("Some-Other-Model") == "some-other-model"
    assert normalise_model(None) == ""


# --- cost arithmetic ---------------------------------------------------------
def test_cost_uses_cached_rate_for_cached_tokens():
    """Cached input must cost strictly less than fresh input."""
    fresh = CostBreakdown(input_tokens=1_000_000, cached_tokens=0, output_tokens=0)
    cached = CostBreakdown(input_tokens=1_000_000, cached_tokens=1_000_000, output_tokens=0)
    assert fresh.usd("MiniMax-M3") == pytest.approx(0.30)
    assert cached.usd("MiniMax-M3") == pytest.approx(0.06)
    assert cached.usd("MiniMax-M3") < fresh.usd("MiniMax-M3")


def test_fresh_input_tokens_excludes_cache():
    b = CostBreakdown(input_tokens=1000, cached_tokens=400, output_tokens=50)
    assert b.fresh_input_tokens == 600


def test_cost_of_unknown_model_is_zero_and_flagged():
    b = CostBreakdown(input_tokens=1000, cached_tokens=0, output_tokens=100)
    assert b.usd("totally-unknown") == 0.0
    d = b.as_dict("totally-unknown")
    assert d["pricing_key"] == "totally-unknown"
    # price_verified defaults True for unknown; as_dict must not invent a rate
    assert d["usd"] == 0.0


def test_flash_preview_price_marked_unverified():
    """The flash price is a placeholder; a report must be able to say so."""
    d = CostBreakdown(input_tokens=1_000_000, cached_tokens=0, output_tokens=0).as_dict(
        "MiniMax-M3.1-Flash-Preview"
    )
    assert d["price_verified"] is False
    assert "PREVIEW" in d["price_note"].upper()


def test_cached_share_pct():
    d = CostBreakdown(input_tokens=1000, cached_tokens=250, output_tokens=0).as_dict("MiniMax-M3")
    assert d["cached_share_pct"] == 25.0


# --- latency percentiles -----------------------------------------------------
def test_p95_never_below_mean_for_the_measured_spread():
    """The engine's core promise: projections use p95, not the optimistic mean."""
    st = _stats("vision", "minimax-m3", [14000, 20000, 45000, 23000, 30000])
    assert st.percentile_ms(95) >= st.mean_ms()


def test_p95_picks_the_high_tail():
    st = _stats("vision_ocr_fusion", "minimax-m3", [19000, 25000, 40000, 50000, 72000])
    # p95 of 5 samples -> the largest
    assert st.percentile_ms(95) == 72000


def test_p95_handles_single_sample():
    st = _stats("vision", "minimax-m3", [12345])
    assert st.percentile_ms(95) == 12345


def test_empty_stats_are_safe():
    st = PathStats(path="vision", model="minimax-m3")
    assert st.percentile_ms() == 0
    assert st.mean_ms() == 0
    assert st.mean_cost() == 0.0
    assert st.mean_cer() is None
    assert st.mean_tokens() == (0, 0, 0)


# --- projection --------------------------------------------------------------
def test_projection_scales_with_records_and_workers():
    st = _stats("vision", "minimax-m3", [20000, 25000, 30000], inp=5000, cached=500, out=2000)
    p = project_records(st, records=1000, workers=1)
    assert p["per_record_ms_p95"] > 0
    assert p["total_cost_usd"] > 0
    assert p["serial_hours"] > p["wall_hours"] or p["workers"] == 1


def test_more_workers_shorten_wall_clock():
    st = _stats("vision", "minimax-m3", [20000] * 5, inp=5000, out=2000)
    one = project_records(st, records=10_000, workers=1)
    eight = project_records(st, records=10_000, workers=8)
    assert eight["wall_hours"] == pytest.approx(one["wall_hours"] / 8, rel=0.01)


def test_projection_cost_independent_of_worker_count():
    """Splitting work across more workers must not change total $."""
    st = _stats("vision", "minimax-m3", [20000] * 4, inp=5000, out=2000)
    a = project_records(st, records=25_000, workers=1)["total_cost_usd"]
    b = project_records(st, records=25_000, workers=8)["total_cost_usd"]
    assert a == b


def test_speedup_needed_reports_infeasible_honestly():
    st = _stats("vision_ocr_fusion", "minimax-m3", [70000] * 6, inp=9000, out=5000)
    p = project_records(st, records=25_000, workers=8)
    s = speedup_needed(p, target_hours=8.0, workers_available=8)
    # 70s * 25000 / 6 samples is enormous; with 8 workers it cannot fit in 8h
    assert s["feasible"] is False
    assert "needs" in s["verdict"]


def test_speedup_needed_feasible_case():
    st = _stats("vision", "minimax-m3", [5000] * 6, inp=5000, out=2000)
    p = project_records(st, records=1000, workers=8)
    s = speedup_needed(p, target_hours=8.0, workers_available=8)
    assert s["feasible"] is True


# --- storage -----------------------------------------------------------------
def test_storage_projection_matches_measured_page_size():
    st = project_storage(25_000)
    assert st["total_gb"] > 50  # measured 2.07 MB/page
    assert st["records"] == 25_000


def test_bytes_per_page_is_positive():
    assert BYTES_PER_PAGE_IMAGE > 1_000_000


# --- aggregate helpers -------------------------------------------------------
def test_mean_cer_ignores_missing_values():
    st = PathStats(path="vision", model="minimax-m3")
    st.add(RunSample(path="vision", model="m", cer=10.0, elapsed_ms=1000))
    st.add(RunSample(path="vision", model="m", cer=None, elapsed_ms=1000))
    st.add(RunSample(path="vision", model="m", cer=20.0, elapsed_ms=1000))
    assert st.mean_cer() == 15.0


def test_usable_ocr_counts_scored_runs():
    st = PathStats(path="tesseract_llm", model="minimax-m3")
    st.add(RunSample(path="tesseract_llm", model="m", score=20.0, ok=True))
    st.add(RunSample(path="tesseract_llm", model="m", score=None, ok=False))
    assert st.usable_ocr() == 1