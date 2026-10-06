"""Shared template helpers for routers."""
from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def render(request: Request, template: str, context: dict | None = None):
    context = context or {}
    context.setdefault("request", request)
    return templates.TemplateResponse(request, template, context)


def pagination(page: int, per_page: int, total: int) -> dict:
    pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, pages))
    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": pages,
        "start": (page - 1) * per_page + 1 if total else 0,
        "end": min(page * per_page, total),
        "has_prev": page > 1,
        "has_next": page < pages,
    }


def qs_with(request: Request, **updates) -> str:
    params = dict(request.query_params)
    params.update({k: str(v) for k, v in updates.items()})
    return "?" + "&".join(f"{k}={v}" for k, v in params.items() if v is not None)


def human_bytes(size: int | None) -> str:
    if not size:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TB"
