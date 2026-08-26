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


_CT_WINDOWS = {
    "abdomen": (40.0, 400.0),
    "liver": (60.0, 150.0),
    "lung": (-600.0, 1500.0),
    "bone": (300.0, 2000.0),
    "brain": (40.0, 80.0),
}


def to_uint8_rgb(slice_data: np.ndarray, window: str = "minmax") -> Image.Image:
    """Scale a 2D slice to uint8 RGB.

    ``window="minmax"`` stretches the slice. Named CT windows clip to
    HU level/width (abdomen, liver, lung, bone, brain).
    """
    data = np.asarray(slice_data, dtype=np.float32)
    spec = _CT_WINDOWS.get(window or "minmax")
    if spec is None:
        minimum = float(data.min()) if data.size else 0.0
        maximum = float(data.max()) if data.size else 0.0
        if maximum > minimum:
            scaled = ((data - minimum) / (maximum - minimum) * 255).astype(np.uint8)
        else:
            scaled = np.zeros(data.shape, dtype=np.uint8)
    else:
        level, width = spec
        lo = level - width / 2.0
        hi = level + width / 2.0
        clipped = np.clip(data, lo, hi)
        scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    if scaled.ndim != 2:
        scaled = np.squeeze(scaled)
    return Image.fromarray(scaled, mode="L").convert("RGB")
