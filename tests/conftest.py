"""Shared pytest fixtures for tests."""

import io

import pytest
from PIL import Image


@pytest.fixture
def sample_image_bytes():
    """Create a sample test image as bytes."""
    img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def sample_png_image_bytes():
    """Create a sample PNG test image as bytes."""
    img = Image.new("RGB", (200, 200), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
