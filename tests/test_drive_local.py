"""Local Drive mirror client tests.

The local mirror is the default provider for Google Drive files; it lets the
pipeline run end-to-end on a developer's machine without OAuth. The real
Google provider uses the same `DriveClient` protocol and is exercised in
`test_google_forms.py`-style live tests (gated on credentials).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.pipeline.ingestion.google_drive import (
    LocalMirrorClient,
    sha256_of,
)


@pytest.fixture
def mirror(tmp_path) -> tuple[LocalMirrorClient, Path]:
    root = tmp_path / "drive_mirror"
    cache = tmp_path / "cache"
    return LocalMirrorClient(root, cache), root


@pytest.mark.asyncio
async def test_local_mirror_lists_files(mirror: tuple[LocalMirrorClient, Path]) -> None:
    client, root = mirror
    (root / "file1.jpg").write_bytes(b"hello")
    (root / "file2.png").write_bytes(b"world")
    files = await client.list_files()
    names = sorted(f.name for f in files)
    assert names == ["file1.jpg", "file2.png"]
    assert all(f.size_bytes > 0 for f in files)


@pytest.mark.asyncio
async def test_local_mirror_downloads_to_dest(mirror: tuple[LocalMirrorClient, Path]) -> None:
    client, root = mirror
    src = root / "source.pdf"
    payload = b"%PDF-fake-bytes-for-the-test"
    src.write_bytes(payload)
    files = await client.list_files()
    target = root.parent / "cache" / "downloaded.pdf"
    size = await client.download(files[0], target)
    assert size == len(payload)
    assert target.read_bytes() == payload


@pytest.mark.asyncio
async def test_local_mirror_handles_empty_folder(mirror: tuple[LocalMirrorClient, Path]) -> None:
    client, _ = mirror
    assert await client.list_files() == []


@pytest.mark.asyncio
async def test_local_mirror_mime_guessing(mirror: tuple[LocalMirrorClient, Path]) -> None:
    client, root = mirror
    for ext in ("jpg", "png", "pdf", "webp"):
        (root / f"file.{ext}").write_bytes(b"x")
    files = await client.list_files()
    mimes = {Path(f.name).suffix: f.mime for f in files}
    assert mimes[".jpg"] == "image/jpeg"
    assert mimes[".png"] == "image/png"
    assert mimes[".pdf"] == "application/pdf"
    assert mimes[".webp"] == "image/webp"


def test_sha256_of(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"