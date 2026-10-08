"""Google Drive ingestion.

Two providers, same `DriveClient` protocol:

  * "local" — uses a folder on disk as a Drive substitute. This is the
    recommended default for local testing: drop the Drive files into
    `./data/drive_mirror/` (or wherever GOOGLE_DRIVE_LOCAL_ROOT points) and the
    importer treats each file as if it were a Drive file, downloading via the
    local copy.

  * "google" — uses the Google Drive v3 `files.get?alt=media` endpoint with the
    OAuth credentials from `google_credentials`. Same scopes the existing Google
    connector uses, plus `drive.readonly`.

Both providers return a `DriveFile` row + a downloaded local copy under
`./data/drive_cache/`.
"""
from __future__ import annotations

import asyncio
import hashlib
import shutil as _shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

from app.config import Settings
from app.models import DriveFile

logger = structlog.get_logger("exam_eval.drive")


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
# local mirror (the local-only testing path)
# ---------------------------------------------------------------------------
class LocalMirrorClient:
    """Use a folder of files as the Drive. Each filename is treated as the
    drive file id (without extension) when no Drive link is available.

    For Drive-link imports, the importer copies the file into this folder first
    (using the file id as the name), so subsequent scans stay consistent.
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


# ---------------------------------------------------------------------------
# Google Drive API client
# ---------------------------------------------------------------------------
class GoogleDriveClient:
    name = "google"

    def __init__(self, settings: Settings, token_loader):
        import httpx
        from googleapiclient.discovery import build

        creds = token_loader()
        self._creds = creds
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._http = httpx

    async def list_files(self, parent_folder: str | None = None) -> list[DriveFileMeta]:
        q = []
        if parent_folder:
            q.append(f"'{parent_folder}' in parents")
        q.append("trashed = false")
        query = " and ".join(q)
        def _list() -> list[DriveFileMeta]:
            results: list[DriveFileMeta] = []
            page_token = None
            while True:
                resp = (
                    self._service.files()
                    .list(
                        q=query or None,
                        pageSize=200,
                        fields="nextPageToken, files(id, name, mimeType, size, parents, webContentLink)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                for f in resp.get("files", []):
                    results.append(
                        DriveFileMeta(
                            drive_file_id=f["id"],
                            name=f.get("name") or "untitled",
                            mime=f.get("mimeType"),
                            size_bytes=int(f.get("size") or 0),
                            parent_folder=(f.get("parents") or [None])[0],
                            download_url=f.get("webContentLink"),
                        )
                    )
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            return results
        return await asyncio.to_thread(_list)

    async def download(self, meta: DriveFileMeta, dest: Path) -> int:
        from googleapiclient.http import MediaIoBaseDownload

        def _download() -> int:
            request = self._service.files().get_media(fileId=meta.drive_file_id)
            fh = dest.open("wb")
            try:
                downloader = MediaIoBaseDownload(fh, request, chunksize=64 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            finally:
                fh.close()
            return dest.stat().st_size

        return await asyncio.to_thread(_download)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_drive_client(settings: Settings, *, token_loader=None) -> DriveClient:
    if settings.google_drive_provider == "google":
        if token_loader is None:
            raise ValueError("token_loader required for Google Drive provider")
        return GoogleDriveClient(settings, token_loader)
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
    """Download a Drive file to local cache and return (or create) a DriveFile row."""

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