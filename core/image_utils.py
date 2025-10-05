from __future__ import annotations

import base64
import io
from typing import Tuple

from PIL import Image

DEFAULT_MAX_IMAGE_SIZE = 1920
DEFAULT_JPEG_QUALITY = 90


def resize_image(image: Image.Image, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> Image.Image:
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = max(1, int(height * (max_size / max(width, 1))))
    else:
        new_height = max_size
        new_width = max(1, int(width * (max_size / max(height, 1))))
    return image.resize((new_width, new_height), Image.LANCZOS)


def image_to_base64(image: Image.Image, quality: int = DEFAULT_JPEG_QUALITY) -> Tuple[str, int]:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG", quality=int(quality), optimize=True)
    raw = img_byte_arr.getvalue()
    return base64.b64encode(raw).decode("utf-8"), len(raw)

