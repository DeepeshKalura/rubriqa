"""Drive ingestion (local mirror only).

For the dev phase, the only supported provider is `local`: drop the Drive
files into `./data/drive_mirror/` and the importer treats each file as a
downloadable Drive file. The protocol shape is preserved so a real provider
can be re-added behind the same interface when the dev phase ends.
"""
from __future__ import annotations

import asyncio
import hashlib
import shutil as _shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog

from app.config import Settings
from app.models import DriveFile

logger = structlog.get_logger("entry_eval.drive")


# ---------------------------------------------------------------------------
# client protocol
# ---------------------------------------------------------------------------
@dataclass
class DriveFileMeta:
    drive_file_id: str
    name: str
    mime: str | None
    size_bytes: int
    parent_folder: str | None = None
    download_url: str | None = None


class DriveClient(Protocol):
    async def list_files(self, parent_folder: str | None = None) -> list[DriveFileMeta]:
        ...

    async def download(self, meta: DriveFileMeta, dest: Path) -> int:
        ...


# ---------------------------------------------------------------------------
# local mirror (the only supported provider in the dev phase)
# ---------------------------------------------------------------------------
class LocalMirrorClient:
    """Use a folder of files as the Drive. Each filename's stem is the
    drive file id; the extension is the file type.
    """

    def __init__(self, root: Path, cache: Path):
        self.root = Path(root)
        self.cache = Path(cache)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)

    async def list_files(self, parent_folder: str | None = None) -> list[DriveFileMeta]:
        results: list[DriveFileMeta] = []
        for p in sorted(self.root.glob("*")):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except FileNotFoundError:
                continue
            results.append(
                DriveFileMeta(
                    drive_file_id=p.stem,
                    name=p.name,
                    mime=_guess_mime(p),
                    size_bytes=size,
                    parent_folder=parent_folder,
                    download_url=str(p),
                )
            )
        return results

    async def download(self, meta: DriveFileMeta, dest: Path) -> int:
        src = Path(meta.download_url or (self.root / meta.name))
        if not src.exists():
            raise FileNotFoundError(f"Drive mirror file missing: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_shutil.copyfile, src, dest)
        return dest.stat().st_size


def _guess_mime(p: Path) -> str:
    suffix = p.suffix.lower()
    return {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".pdf":  "application/pdf",
        ".webp": "image/webp",
        ".tif":  "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")


def build_drive_client(settings: Settings, *, token_loader=None) -> DriveClient:
    cache = settings.storage_root / "drive_cache"
    return LocalMirrorClient(settings.google_drive_local_root, cache)


# ---------------------------------------------------------------------------
# helpers used by other modules
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def mirror_drive_url_to_local(
    client: DriveClient,
    settings: Settings,
    drive_url: str,
    drive_file_id: str,
    name_hint: str,
) -> DriveFile:
    """Mirror a drive file URL to the local cache and return a DriveFile row."""
    cache_dir = settings.storage_root / "drive_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(name_hint).suffix or ""
    target = cache_dir / f"{drive_file_id}{ext}"
    if not target.exists():
        meta = DriveFileMeta(
            drive_file_id=drive_file_id,
            name=name_hint,
            mime=_guess_mime(Path(name_hint)),
            size_bytes=0,
            download_url=drive_url,
        )
        try:
            await client.download(meta, target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("drive_download_failed", file_id=drive_file_id, error=str(exc)[:300])
            return DriveFile(
                drive_file_id=drive_file_id,
                name=name_hint,
                mime=meta.mime,
                size_bytes=0,
                download_url=drive_url,
                status="failed",
                error=str(exc)[:500],
            )

    sha = sha256_of(target)
    return DriveFile(
        drive_file_id=drive_file_id,
        name=name_hint,
        mime=_guess_mime(target),
        size_bytes=target.stat().st_size,
        sha256=sha,
        local_path=str(target),
        download_url=drive_url,
        fetched_at=None,
        status="fetched",
    )
