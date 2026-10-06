"""File serving for derived artifacts (page images/thumbnails) and originals."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.storage import Storage

router = APIRouter(tags=["files"])

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
}


@router.get("/files/{path:path}")
async def serve_file(path: str):
    storage = Storage()
    candidate = storage.resolve(path)
    if candidate is None:
        raise HTTPException(404, "File not found")
    resolved = Path(candidate).resolve()
    root = storage.root.resolve()
    # path traversal protection: only serve files inside the storage root
    if not str(resolved).startswith(str(root)) or not resolved.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(resolved, media_type=MIME_TYPES.get(resolved.suffix.lower()))
