"""Unit tests for core redaction logic."""

import numpy as np
import pytest
import cv2

from src.redact_id.core import (
    Detection,
    redact_image_bgr,
    detections_from_ultralytics_result,
    _clip_box,
    _auto_kernel_from_box,
    FULL_BLUR_CLASSES,
    PARTIAL_BLUR_CLASSES,
)


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        """Test creating a Detection instance."""
        det = Detection(cls_id=0, conf=0.95, xyxy=(10, 20, 100, 200))
        assert det.cls_id == 0
        assert det.conf == 0.95
        assert det.xyxy == (10, 20, 100, 200)

    def test_detection_immutable(self):
        """Test that Detection is immutable."""
        det = Detection(cls_id=0, conf=0.95, xyxy=(10, 20, 100, 200))
        with pytest.raises(Exception):
            det.cls_id = 1


class TestUtilities:
    """Tests for utility functions."""

    def test_clip_box_within_bounds(self):
        """Test clipping box that's within image bounds."""
        result = _clip_box(10, 20, 100, 200, w=500, h=500)
        assert result == (10, 20, 100, 200)

    def test_clip_box_outside_bounds(self):
        """Test clipping box that exceeds image bounds."""
        result = _clip_box(-10, -20, 600, 700, w=500, h=500)
        assert result == (0, 0, 500, 500)

    def test_clip_box_invalid(self):
        """Test invalid box returns None."""
        result = _clip_box(100, 200, 50, 100, w=500, h=500)
        assert result is None

    def test_auto_kernel_from_box(self):
        """Test automatic kernel size selection."""
        k = _auto_kernel_from_box(100, 100)
        assert isinstance(k, tuple)
        assert len(k) == 2
        assert k[0] % 2 == 1  # Must be odd
        assert k[1] % 2 == 1

    def test_auto_kernel_small_box(self):
        """Test kernel for small box."""
        k = _auto_kernel_from_box(10, 10)
        assert k[0] >= 21  # Minimum size


class TestRedactImageBgr:
    """Tests for core redaction function."""

    def test_redact_empty_detections(self):
        """Test redaction with no detections."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (255, 255, 255)  # White image

        result = redact_image_bgr(img, [])
        np.testing.assert_array_equal(result, img)

    def test_redact_full_blur_class(self):
        """Test full blur for mandatory classes."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create gradient pattern so blur has visible effect
        for i in range(100):
            img[i, :] = (i * 2, 255 - i * 2, 128)

        original_region = img[10:50, 10:50].copy()

        # Class 0 (Address) should be fully blurred
        det = Detection(cls_id=0, conf=0.95, xyxy=(10, 10, 50, 50))
        result = redact_image_bgr(img, [det])

        # Check that the region was modified
        assert not np.array_equal(result[10:50, 10:50], original_region)

    def test_redact_partial_blur_class_single(self):
        """Test partial blur for single detection in partial group."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create gradient
        for i in range(100):
            for j in range(100):
                img[i, j] = (i, j, 128)

        original_region = img[10:50, 10:50].copy()

        # Class 2 (ID Number) should be partially blurred when alone
        det = Detection(cls_id=2, conf=0.95, xyxy=(10, 10, 50, 50))
        result = redact_image_bgr(img, [det])

        # Region should be modified
        assert not np.array_equal(result[10:50, 10:50], original_region)

    def test_redact_partial_blur_class_multiple_best_conf(self):
        """Test partial blur chooses highest confidence detection."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create gradient with variation
        for i in range(100):
            for j in range(100):
                img[i, j] = (i * 2 % 256, j * 2 % 256, (i + j) % 256)

        orig1 = img[10:30, 10:30].copy()
        orig2 = img[40:60, 40:60].copy()
        orig3 = img[70:90, 70:90].copy()

        # Multiple partial group detections, highest conf should get partial blur
        det1 = Detection(cls_id=2, conf=0.85, xyxy=(10, 10, 30, 30))
        det2 = Detection(cls_id=4, conf=0.95, xyxy=(40, 40, 60, 60))  # Best
        det3 = Detection(cls_id=6, conf=0.75, xyxy=(70, 70, 90, 90))

        result = redact_image_bgr(img, [det1, det2, det3])

        # All regions should be modified
        assert not np.array_equal(result[10:30, 10:30], orig1)
        assert not np.array_equal(result[40:60, 40:60], orig2)
        assert not np.array_equal(result[70:90, 70:90], orig3)

    def test_redact_mixed_classes(self):
        """Test redaction with mixed class types."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create gradient
        for i in range(200):
            for j in range(200):
                img[i, j] = (i % 256, j % 256, (i + j) // 2 % 256)

        originals = {}
        dets = [
            Detection(cls_id=0, conf=0.95, xyxy=(10, 10, 50, 50)),  # Full blur
            Detection(cls_id=2, conf=0.90, xyxy=(60, 60, 100, 100)),  # Partial
            Detection(cls_id=1, conf=0.85, xyxy=(110, 110, 150, 150)),  # Full blur
        ]

        for det in dets:
            x1, y1, x2, y2 = det.xyxy
            originals[det] = img[y1:y2, x1:x2].copy()

        result = redact_image_bgr(img, dets)

        # All regions should be modified
        for det in dets:
            x1, y1, x2, y2 = det.xyxy
            assert not np.array_equal(result[y1:y2, x1:x2], originals[det])

    def test_redact_creates_copy(self):
        """Test that redaction creates a copy and doesn't modify original."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create gradient
        for i in range(100):
            img[i, :] = (i * 2, 255 - i * 2, 128)

        original_copy = img.copy()

        det = Detection(cls_id=0, conf=0.95, xyxy=(10, 10, 50, 50))
        result = redact_image_bgr(img, [det])

        # Original should be unchanged
        np.testing.assert_array_equal(img, original_copy)
        # Result should be different
        assert not np.array_equal(result, img)


class TestDetectionsFromUltralyticsResult:
    """Tests for Ultralytics result conversion."""

    def test_none_result(self):
        """Test handling None result."""
        dets = detections_from_ultralytics_result(None)
        assert dets == []

    def test_result_without_boxes(self):
        """Test handling result without boxes."""
        class MockResult:
            boxes = None

        dets = detections_from_ultralytics_result(MockResult())
        assert dets == []

    def test_valid_result(self):
        """Test conversion of valid Ultralytics result."""
        class MockBoxes:
            def __init__(self):
                import torch
                self.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
                self.cls = torch.tensor([0.0])
                self.conf = torch.tensor([0.95])

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes()

        dets = detections_from_ultralytics_result(MockResult())

        assert len(dets) == 1
        assert dets[0].cls_id == 0
        assert abs(dets[0].conf - 0.95) < 0.01  # Allow small float precision difference
        assert dets[0].xyxy == (10, 20, 100, 200)


class TestConstants:
    """Tests for configuration constants."""

    def test_full_blur_classes(self):
        """Test full blur classes are defined correctly."""
        assert 0 in FULL_BLUR_CLASSES  # Address
        assert 1 in FULL_BLUR_CLASSES  # Birth Date
        assert 3 in FULL_BLUR_CLASSES  # Long Passport Number

    def test_partial_group_classes(self):
        """Test partial group classes are defined correctly."""
        assert 2 in PARTIAL_BLUR_CLASSES  # ID Number
        assert 4 in PARTIAL_BLUR_CLASSES  # NHI ID
        assert 6 in PARTIAL_BLUR_CLASSES  # Passport Number

    def test_classes_mutually_exclusive(self):
        """Test that full and partial blur classes don't overlap."""
        assert FULL_BLUR_CLASSES.isdisjoint(PARTIAL_BLUR_CLASSES)
