"""Preprocessing: page rendering, deskew, denoise, upscale, thumbnails.

Kept intentionally small: PIL + numpy only for the core path so it stays fast
on a mid-range laptop. The functions are pure file-in/file-out operations and
are safe to call from threads.
"""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import numpy as np
from PIL import Image, ImageFilter, ImageOps

RENDER_DPI = 220  # good balance for OCR/handwriting on an office laptop


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int = RENDER_DPI) -> Image.Image:
    """Render a single PDF page to a PIL image at the given DPI."""
    with pymupdf.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return image


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def deskew_score(image: Image.Image) -> float:
    """Estimate skew angle in degrees using row-darkness variance.

    Cheap projection profile method: rotate in small steps, pick the angle that
    maximizes the variance of horizontal ink projections (text lines align).
    """
    gray = image.convert("L").resize((min(image.width, 1000), min(image.height, 1000)))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-4, 4.5, 1.0):
        rotated = gray.rotate(float(angle), resample=Image.BILINEAR, fillcolor=255)
        rot_arr = 255 - np.asarray(rotated, dtype=np.float32)
        score = float(np.var(rot_arr.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def preprocess_for_ocr(image: Image.Image, max_side: int = 2200) -> Image.Image:
    """Grayscale, deskew, denoise, contrast-normalize, upscale small pages."""
    gray = image.convert("L")
    if max(gray.size) < 1200:
        factor = min(2.0, 1200 / max(gray.size))
        gray = gray.resize((int(gray.width * factor), int(gray.height * factor)), Image.LANCZOS)
    angle = deskew_score(gray)
    if abs(angle) >= 0.5:
        gray = gray.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = ImageOps.autocontrast(gray, cutoff=1)
    if max(gray.size) > max_side:
        scale = max_side / max(gray.size)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)
    return gray


def to_jpeg_bytes(image: Image.Image, max_side: int = 2200, quality: int = 85) -> bytes:
    img = image
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def save_page_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    image.save(tmp, format="PNG", optimize=True)
    tmp.replace(path)


def save_thumbnail(image: Image.Image, path: Path, max_side: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb = image.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    if thumb.mode != "RGB":
        thumb = thumb.convert("RGB")
    tmp = path.with_suffix(".tmp.jpg")
    thumb.save(tmp, format="JPEG", quality=80)
    tmp.replace(path)
