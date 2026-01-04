"""Integration tests for FastAPI endpoints."""

import io
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.redact_id.api import create_app


@pytest.fixture
def mock_model():
    """Create a mock YOLO model."""
    model = Mock()

    # Mock result
    mock_result = Mock()
    mock_boxes = Mock()

    # Create mock tensors
    import torch
    mock_boxes.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    mock_boxes.cls = torch.tensor([0.0])
    mock_boxes.conf = torch.tensor([0.95])

    mock_result.boxes = mock_boxes
    model.predict.return_value = [mock_result]

    return model


@pytest.fixture
def client(mock_model, tmp_path):
    """Create test client with mocked model."""
    # Create a dummy model file
    model_path = str(tmp_path / "test_model.pt")
    with open(model_path, "w") as f:
        f.write("dummy")

    # Patch YOLO to return our mock
    with patch("src.redact_id.api.YOLO", return_value=mock_model):
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def test_image_bytes():
    """Create a test image as bytes."""
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "RedactID"


class TestRedactEndpoint:
    """Tests for redaction endpoint."""

    def test_redact_valid_image(self, client, test_image_bytes):
        """Test redacting a valid image."""
        response = client.post(
            "/redact",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0

    def test_redact_with_custom_params(self, client, test_image_bytes):
        """Test redaction with custom parameters."""
        response = client.post(
            "/redact",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            params={"conf_threshold": 0.5, "keep_ratio": 0.4},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_redact_invalid_file_type(self, client):
        """Test uploading non-image file."""
        response = client.post(
            "/redact",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"].lower()

    def test_redact_invalid_conf_threshold(self, client, test_image_bytes):
        """Test with invalid confidence threshold."""
        response = client.post(
            "/redact",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            params={"conf_threshold": 1.5},  # Invalid: > 1.0
        )

        assert response.status_code == 422  # Validation error

    def test_redact_invalid_keep_ratio(self, client, test_image_bytes):
        """Test with invalid keep ratio."""
        response = client.post(
            "/redact",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            params={"keep_ratio": 1.5},  # Invalid: > 0.95
        )

        assert response.status_code == 422  # Validation error

    def test_redact_corrupted_image(self, client):
        """Test with corrupted image data."""
        response = client.post(
            "/redact",
            files={"file": ("test.jpg", b"corrupted data", "image/jpeg")},
        )

        assert response.status_code == 400

    def test_redact_empty_file(self, client):
        """Test with empty file."""
        response = client.post(
            "/redact",
            files={"file": ("test.jpg", b"", "image/jpeg")},
        )

        # Empty file causes decoding error, returns 400 or 500
        assert response.status_code in [400, 500]

    def test_redact_png_image(self, client):
        """Test redacting PNG image."""
        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        png_bytes = buf.getvalue()

        response = client.post(
            "/redact",
            files={"file": ("test.png", png_bytes, "image/png")},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_redact_response_headers(self, client, test_image_bytes):
        """Test response headers are set correctly."""
        response = client.post(
            "/redact",
            files={"file": ("document.jpg", test_image_bytes, "image/jpeg")},
        )

        assert response.status_code == 200
        assert "content-disposition" in response.headers
        assert "redacted_document.jpg" in response.headers["content-disposition"]


class TestAsyncBehavior:
    """Tests for async behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, test_image_bytes):
        """Test handling concurrent requests."""
        from concurrent.futures import ThreadPoolExecutor

        def make_request():
            return client.post(
                "/redact",
                files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            )

        # Make 5 concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_file_parameter(self, client):
        """Test missing file parameter."""
        response = client.post("/redact")
        assert response.status_code == 422  # Validation error

    def test_model_not_loaded(self, tmp_path):
        """Test behavior when model fails to load."""

        with patch("src.redact_id.api.YOLO", side_effect=Exception("Model load failed")):
            with pytest.raises(Exception, match="Model load failed"):
                app = create_app()
                # Trigger lifespan startup which should raise the exception
                with TestClient(app):
                    pass
