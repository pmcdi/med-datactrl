"""Image helpers for LLM-ready payloads."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def to_llm_image_payload(
    *,
    id: str,
    kind: str,
    image: Image.Image,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode a PIL image as a base64 PNG payload for tool responses.

    Args:
        id: Resource or preview id.
        kind: Payload kind label (e.g. ``dicom``).
        image: Source image to encode.
        meta: Optional metadata merged into the payload.

    Returns:
        Dict with ``image_base64``, ``mime_type``, and ``meta``.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "id": id,
        "kind": kind,
        "image_base64": encoded,
        "mime_type": "image/png",
        "meta": meta or {},
    }


def to_uint8_rgb(slice_data: np.ndarray) -> Image.Image:
    """Min-max scale a 2D slice to uint8 RGB for preview encoding."""
    data = slice_data.astype(np.float32)
    minimum = float(data.min())
    maximum = float(data.max())

    if maximum > minimum:
        data = (data - minimum) / (maximum - minimum)
        scaled = (data * 255).astype(np.uint8)
    else:
        scaled = np.zeros(data.shape, dtype=np.uint8)

    return Image.fromarray(scaled, mode="L").convert("RGB")
