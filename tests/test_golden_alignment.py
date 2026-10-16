"""The golden transcript set must stay aligned with the page images.

This exists because it silently was not. On 2026-10-14 the transcripts for
images 08/09/10 were filed as page-07/08/09, and images 07 and 11 never had a
transcript at all. Nothing failed: the benchmark ran, printed a mean CER, and
that number went into a decision record and a GitHub issue. It was only caught
because a competent transcription was cross-correlated against *every* golden
file and one pairing came back at 0.3% while the nominal one came back at
209%.

So: goldens are resolved through `data/samples/golden/manifest.json`, and these
tests assert that indirection stays sound.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# The live bind mount wins over the image's baked copy (docker-compose.yml
# mounts ./data:/data), matching the bench scripts.
_SAMPLES = Path("/data/samples") if Path("/data/samples/pages").is_dir() else Path("data/samples")
MANIFEST = _SAMPLES / "golden" / "manifest.json"
PAGES = _SAMPLES / "pages"


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} not present in this environment")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_exists_and_declares_a_map():
    manifest = _manifest()
    assert manifest.get("map"), "manifest must declare the image -> golden map"
    assert "scored_pages" in manifest


def test_every_mapped_golden_file_exists_and_is_not_empty():
    for image, filename in _manifest()["map"].items():
        path = MANIFEST.parent / filename
        assert path.exists(), f"{image} maps to missing {filename}"
        assert path.read_text(encoding="utf-8").strip(), f"{filename} is empty"


def test_every_mapped_image_exists():
    for image in _manifest()["map"]:
        assert (PAGES / f"{image}.png").exists(), f"manifest maps missing image {image}"


def test_no_golden_is_claimed_by_two_images():
    """The exact failure mode: one transcript serving two images would let a
    misalignment hide as a plausible number."""
    targets = list(_manifest()["map"].values())
    assert len(targets) == len(set(targets)), f"duplicate golden targets: {targets}"


def test_images_without_golden_are_declared_explicitly():
    """An image with no transcript must be a stated fact, not a gap. Silent
    gaps are how two pages dropped out of the average without anyone noticing."""
    manifest = _manifest()
    declared = set(manifest.get("images_without_golden", {}))
    mapped = set(manifest["map"])
    every_image = {p.stem for p in PAGES.glob("page-*.png")}
    assert mapped | declared == every_image, (
        f"images neither mapped nor declared: {sorted(every_image - mapped - declared)}"
    )
    assert not (mapped & declared), f"image both mapped and declared missing: {mapped & declared}"


def test_scored_pages_matches_the_map():
    manifest = _manifest()
    assert manifest["scored_pages"] == len(manifest["map"])


def test_bench_script_refuses_to_run_without_a_manifest(tmp_path, monkeypatch):
    """The guard that stops a future rename from quietly reintroducing this."""
    import scripts.bench_cloud_ocr as bench

    monkeypatch.setattr(bench, "GOLDEN_DIR", tmp_path)
    with pytest.raises(SystemExit, match="manifest"):
        bench.load_manifest()


def test_golden_for_returns_none_for_an_image_with_no_transcript():
    import scripts.bench_cloud_ocr as bench

    manifest = _manifest()
    missing = sorted(
        set(manifest.get("images_without_golden", {})) - set(manifest["map"])
    )
    if not missing:
        pytest.skip("every image has a transcript")
    assert bench.golden_for(Path(f"{missing[0]}.png"), manifest) is None


def test_golden_for_resolves_through_the_manifest_not_the_filename():
    import scripts.bench_cloud_ocr as bench

    manifest = _manifest()
    image, filename = next(iter(manifest["map"].items()))
    expected = (bench.GOLDEN_DIR / filename).read_text(encoding="utf-8")
    assert bench.golden_for(Path(f"{image}.png"), manifest) == expected

# --- the sandbox endpoint must obey the same rule ---------------------------


def test_sandbox_resolves_goldens_through_the_manifest(tmp_path):
    """The compare endpoint used to look goldens up by filename, which is how a
    209% CER reached the UI. It must go through the manifest too."""
    from app.routers.sandbox import _golden_transcripts

    manifest = _manifest()
    samples = tmp_path / "samples"
    (samples / "pages").mkdir(parents=True)
    (samples / "golden").mkdir(parents=True)

    # Reconstruct a deliberately mislabelled set: the image "page-08" is
    # declared to use the file "renamed.txt", and a same-named decoy exists.
    import shutil

    src = MANIFEST.parent / manifest["map"]["page-08"]
    (samples / "golden" / "page-08.txt").write_text(
        "DECOY - this file only exists to be matched by filename", encoding="utf-8"
    )
    (samples / "golden" / "renamed.txt").write_text(
        src.read_text(encoding="utf-8"), encoding="utf-8"
    )
    shutil.copyfile(
        next(iter(sorted((MANIFEST.parent.parent / "pages").glob("page-08.png")))),
        samples / "pages" / "page-08.png",
    )
    (samples / "golden" / "manifest.json").write_text(
        json.dumps({"map": {"page-08": "renamed.txt"}}), encoding="utf-8"
    )

    class _S:
        storage_root = tmp_path

    got = _golden_transcripts(_S())
    assert len(got) == 1
    transcript = next(iter(got.values()))
    assert "DECOY" not in transcript
    assert transcript == src.read_text(encoding="utf-8").strip()


def test_sandbox_returns_no_transcripts_when_the_manifest_is_missing(tmp_path):
    from app.routers.sandbox import _golden_transcripts

    samples = tmp_path / "samples"
    (samples / "pages").mkdir(parents=True)
    (samples / "golden").mkdir(parents=True)
    (samples / "golden" / "page-01.txt").write_text("orphan transcript", encoding="utf-8")
    (samples / "pages" / "page-01.png").write_bytes(b"x")

    class _S:
        storage_root = tmp_path

    # No manifest: nothing is claimed, rather than everything being matched by
    # the accident that names line up.
    assert _golden_transcripts(_S()) == {}
