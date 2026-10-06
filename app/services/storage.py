"""Content-addressed local storage for original files and derived artifacts.

Layout (inside STORAGE_ROOT, normally ./data):
    originals/<sha[:2]>/<sha>.<ext>       immutable uploaded files
    derived/<document_id>/page_0001.png  rendered page images
    derived/<document_id>/thumb_0001.jpg thumbnails
    exports/                             generated CSV/report exports
    uploads/<batch>/...                  transient upload landing zone
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from app.config import get_settings


class Storage:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or get_settings().storage_root)

    # -- layout helpers ------------------------------------------------------
    @property
    def originals(self) -> Path:
        return self.root / "originals"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    def ensure_dirs(self) -> None:
        for path in (self.originals, self.derived, self.exports, self.uploads):
            path.mkdir(parents=True, exist_ok=True)

    # -- hashing -------------------------------------------------------------
    @staticmethod
    def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    # -- atomic writes -------------------------------------------------------
    @staticmethod
    def atomic_write_bytes(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def store_original(self, source: Path, sha256: str, ext: str) -> Path:
        """Copy a file into content-addressed storage (idempotent)."""
        ext = ext.lstrip(".").lower() or "bin"
        target = self.originals / sha256[:2] / f"{sha256}.{ext}"
        if target.exists() and target.stat().st_size == source.stat().st_size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        return target

    def document_dir(self, document_id: int) -> Path:
        path = self.derived / str(document_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def page_image_path(self, document_id: int, page_index: int) -> Path:
        return self.document_dir(document_id) / f"page_{page_index + 1:04d}.png"

    def page_thumb_path(self, document_id: int, page_index: int) -> Path:
        return self.document_dir(document_id) / f"thumb_{page_index + 1:04d}.jpg"

    def resolve(self, stored: str | None) -> Path | None:
        """Resolve a path stored in the DB (absolute or storage-relative)."""
        if not stored:
            return None
        path = Path(stored)
        if path.is_absolute():
            return path
        return self.root / path

    def safe_resolve(self, requested: str) -> Path | None:
        """Resolve a user-supplied path strictly inside the storage root.

        Used by import endpoints so a request like "../../etc" cannot point
        at arbitrary host directories. Absolute paths are accepted only if
        they already live under the storage root; the result is symlink
        resolved and containment-checked.
        """
        if not requested:
            return None
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        root = self.root.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None
        return resolved

    def relative(self, path: Path) -> str:
        """Store paths relative to the storage root so ./data stays portable."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)
