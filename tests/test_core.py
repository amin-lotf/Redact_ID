# tests/unit/test_core.py

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, List, Tuple

import cv2
import numpy as np
import pytest


# ---------- Helpers to import core safely (MODEL_PATH validated at import time) ----------
def import_core_with_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """
    settings.Settings validates MODEL_PATH exists at import-time.
    So we must set env before importing redact_id.core.
    """
    weights = tmp_path / "dummy_yolo_weights.pt"
    weights.write_bytes(b"not-a-real-model")

    monkeypatch.setenv("MODEL_PATH", str(weights))
    # optional: ensure policy is not required
    monkeypatch.delenv("REDACTION_POLICY_PATH", raising=False)

    # Import/reload settings + core after env is set
    import redact_id.settings as settings_mod
    importlib.reload(settings_mod)

    import redact_id.core as core_mod
    importlib.reload(core_mod)
    return core_mod


# ---------- Minimal tensor-like wrappers (to mimic ultralytics result.boxes.* API) ----------
class _TensorLike:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


@dataclass
class _FakeBoxes:
    xyxy: _TensorLike
    cls: _TensorLike
    conf: _TensorLike


@dataclass
class _FakeResult:
    boxes: _FakeBoxes


class _FakeYOLO:
    """
    A fake YOLO model with a .predict() method returning one _FakeResult.
    """
    def __init__(self, result: _FakeResult):
        self._result = result

    def predict(self, source: Any, conf: float = 0.25, verbose: bool = False):
        return [self._result]


# ---------- Image generators ----------
def make_noise_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return img


def variance_of_region(img: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    # variance across all channels/pixels
    return float(np.var(roi))


# ---------- Tests ----------
def test_resize_if_needed_no_resize(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    resized, scale = core.resize_if_needed(img, max_dim=512)

    assert resized.shape == img.shape
    assert scale == 1.0


def test_resize_if_needed_resizes_proportionally(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    # bigger than max_dim
    img = np.zeros((2000, 1000, 3), dtype=np.uint8)  # h=2000, w=1000
    resized, scale = core.resize_if_needed(img, max_dim=500)

    # scale should be min(500/1000, 500/2000) = 0.25
    assert pytest.approx(scale, rel=1e-6) == 0.25
    assert resized.shape[0] == int(2000 * scale)
    assert resized.shape[1] == int(1000 * scale)


def test_clip_box(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    # image w=100 h=80
    assert core._clip_box(-5, -5, 50, 50, w=100, h=80) == (0, 0, 50, 50)
    assert core._clip_box(10, 10, 200, 90, w=100, h=80) == (10, 10, 100, 80)

    # invalid (x2<=x1 or y2<=y1)
    assert core._clip_box(10, 10, 10, 20, w=100, h=80) is None
    assert core._clip_box(10, 10, 20, 10, w=100, h=80) is None


def test_auto_kernel_is_odd_and_bounded(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    kx, ky = core._auto_kernel_from_box(120, 40)
    assert kx == ky
    assert kx % 2 == 1
    assert 21 <= kx <= 81


def test_blur_region_changes_only_inside_box(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    img = make_noise_image(120, 160, seed=1)
    out = img.copy()

    box = (30, 20, 110, 80)
    outside_box = (0, 0, 20, 20)

    before_inside = out[box[1]:box[3], box[0]:box[2]].copy()
    before_outside = out[outside_box[1]:outside_box[3], outside_box[0]:outside_box[2]].copy()

    core._blur_region(out, box, kernel=(31, 31))

    after_inside = out[box[1]:box[3], box[0]:box[2]]
    after_outside = out[outside_box[1]:outside_box[3], outside_box[0]:outside_box[2]]

    assert not np.array_equal(before_inside, after_inside), "inside ROI should change"
    assert np.array_equal(before_outside, after_outside), "outside ROI should remain identical"


def test_partial_blur_keep_right_blurs_left_keeps_right(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    img = make_noise_image(80, 200, seed=2)
    out = img.copy()

    box = (20, 10, 180, 60)  # wide box
    keep_ratio = 0.25
    bw = box[2] - box[0]
    blur_x2 = int(box[2] - bw * keep_ratio)

    left_box = (box[0], box[1], blur_x2, box[3])
    right_box = (blur_x2, box[1], box[2], box[3])

    before_left_var = variance_of_region(out, left_box)
    before_right = out[right_box[1]:right_box[3], right_box[0]:right_box[2]].copy()

    core._partial_blur_keep_right(out, box, keep_ratio=keep_ratio, kernel=(31, 31))

    after_left_var = variance_of_region(out, left_box)
    after_right = out[right_box[1]:right_box[3], right_box[0]:right_box[2]]

    # blur reduces variance in noisy region
    assert after_left_var < before_left_var
    # right region should remain unchanged
    assert np.array_equal(before_right, after_right)


def test_redact_image_bgr_partial_best_selects_highest_conf(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    # Force deterministic policy sets for this test
    monkeypatch.setattr(core, "FULL_BLUR_CLASSES", {0})
    monkeypatch.setattr(core, "PARTIAL_BLUR_CLASSES", {1})

    img = make_noise_image(120, 220, seed=3)

    # one full-blur, two partial candidates
    det_full = core.Detection(cls_id=0, conf=0.9, xyxy=(10, 10, 80, 50))

    # partial candidates: best is conf=0.95
    det_p1 = core.Detection(cls_id=1, conf=0.80, xyxy=(90, 10, 200, 50))
    det_p2 = core.Detection(cls_id=1, conf=0.95, xyxy=(90, 60, 200, 100))

    out = core.redact_image_bgr(
        img,
        [det_full, det_p1, det_p2],
        keep_ratio=0.3,
        partial_mode="best",
        full_kernel=(31, 31),
        partial_kernel=(31, 31),
    )

    # Full region should change
    assert not np.array_equal(
        img[10:50, 10:80],
        out[10:50, 10:80],
    )

    # For det_p1 (not best): should be fully blurred => entire box changed
    assert not np.array_equal(
        img[10:50, 90:200],
        out[10:50, 90:200],
    )

    # For det_p2 (best): left part blurred, right part preserved
    x1, y1, x2, y2 = det_p2.xyxy
    bw = x2 - x1
    blur_x2 = int(x2 - bw * 0.3)

    left_before = img[y1:y2, x1:blur_x2]
    left_after = out[y1:y2, x1:blur_x2]
    right_before = img[y1:y2, blur_x2:x2]
    right_after = out[y1:y2, blur_x2:x2]

    assert not np.array_equal(left_before, left_after), "best partial: left should blur"
    assert np.array_equal(right_before, right_after), "best partial: right should remain"


def test_get_blur_type(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    monkeypatch.setattr(core, "FULL_BLUR_CLASSES", {0})
    monkeypatch.setattr(core, "PARTIAL_BLUR_CLASSES", {1})

    d_full = core.Detection(0, 0.9, (0, 0, 10, 10))
    d_p1 = core.Detection(1, 0.7, (10, 0, 40, 20))
    d_p2 = core.Detection(1, 0.8, (10, 30, 40, 50))
    d_other = core.Detection(99, 0.5, (0, 0, 5, 5))

    dets = [d_full, d_p1, d_p2, d_other]

    assert core.get_blur_type(d_full, dets, "best") == "full"
    assert core.get_blur_type(d_other, dets, "best") == "unknown"

    # best: only highest conf in partial group is partial
    assert core.get_blur_type(d_p2, dets, "best") == "partial"
    assert core.get_blur_type(d_p1, dets, "best") == "full"

    # all: all partial group are partial
    assert core.get_blur_type(d_p1, dets, "all") == "partial"
    assert core.get_blur_type(d_p2, dets, "all") == "partial"


def test_encode_image_jpeg_roundtrip(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    img = make_noise_image(80, 120, seed=4)
    jpeg = core.encode_image_jpeg(img, quality=90)

    assert isinstance(jpeg, (bytes, bytearray))
    assert len(jpeg) > 100  # should be non-trivial

    nparr = np.frombuffer(jpeg, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == img.shape[:2]


@pytest.mark.asyncio
async def test_redact_and_encode_async_outputs_jpeg(monkeypatch, tmp_path):
    core = import_core_with_env(monkeypatch, tmp_path)

    monkeypatch.setattr(core, "FULL_BLUR_CLASSES", {0})
    monkeypatch.setattr(core, "PARTIAL_BLUR_CLASSES", {1})

    img = make_noise_image(100, 150, seed=5)
    dets = [core.Detection(0, 0.9, (10, 10, 80, 40))]

    out_bytes = await core.redact_and_encode_async(
        img,
        dets,
        keep_ratio=0.3,
        partial_mode="best",
        jpeg_quality=85,
    )

    assert isinstance(out_bytes, (bytes, bytearray))
    nparr = np.frombuffer(out_bytes, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert decoded is not None


@pytest.mark.asyncio
async def test_detect_async_decodes_and_converts_detections(monkeypatch, tmp_path, sample_image_bytes):
    core = import_core_with_env(monkeypatch, tmp_path)

    # Fake one detection
    xyxy = np.array([[10, 20, 110, 60]], dtype=np.float32)
    cls = np.array([2], dtype=np.float32)
    conf = np.array([0.88], dtype=np.float32)

    fake_result = _FakeResult(
        boxes=_FakeBoxes(
            xyxy=_TensorLike(xyxy),
            cls=_TensorLike(cls),
            conf=_TensorLike(conf),
        )
    )
    model = _FakeYOLO(fake_result)

    img, dets, (orig_w, orig_h), scale = await core.detect_async(
        sample_image_bytes,
        model,
        conf_thres=0.25,
    )

    assert img is not None
    assert isinstance(dets, list)
    assert len(dets) == 1
    assert dets[0].cls_id == 2
    assert pytest.approx(dets[0].conf, rel=1e-6) == 0.88
    assert dets[0].xyxy == (10, 20, 110, 60)

    # original dims should match the sample image (200x200 from your fixture)
    assert (orig_w, orig_h) == (200, 200)
    assert 0 < scale <= 1.0
