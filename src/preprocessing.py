from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class PreprocessResult:
    original_path: Path
    processed_path: Path
    width: int
    height: int
    mode: str
    preprocessing_time_sec: float
    background_removed: bool


def validate_image_path(image_path: str | Path, max_file_size_mb: float = 25.0) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {path.suffix}. Supported: PNG, JPG, JPEG, WEBP")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_file_size_mb:
        raise ValueError(f"Image is too large: {size_mb:.2f} MB. Limit: {max_file_size_mb:.2f} MB")
    try:
        with Image.open(path) as image:
            image.verify()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unreadable image: {path}") from exc
    return path


def _resize_preserving_aspect(image: Image.Image, max_size: int) -> Image.Image:
    if max(image.size) <= max_size:
        return image
    resized = image.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return resized


def _remove_background_if_available(image: Image.Image) -> tuple[Image.Image, bool]:
    try:
        from rembg import remove

        return remove(image.convert("RGBA")), True
    except ImportError:
        return image.convert("RGBA"), False


def preprocess_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    remove_background: bool = True,
    max_image_size: int = 1024,
) -> PreprocessResult:
    start = time.perf_counter()
    source = validate_image_path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_copy = out_dir / "input.png"
    if source.resolve() != input_copy.resolve():
        with Image.open(source) as image:
            image.convert("RGBA").save(input_copy)
    else:
        shutil.copy2(source, input_copy)

    with Image.open(source) as image:
        image = image.convert("RGBA")
        image = _resize_preserving_aspect(image, max_image_size)
        background_removed = False
        if remove_background:
            image, background_removed = _remove_background_if_available(image)
        processed_path = out_dir / "processed.png"
        image.save(processed_path)
        width, height = image.size
        mode = image.mode

    return PreprocessResult(
        original_path=input_copy,
        processed_path=processed_path,
        width=width,
        height=height,
        mode=mode,
        preprocessing_time_sec=round(time.perf_counter() - start, 4),
        background_removed=background_removed,
    )
