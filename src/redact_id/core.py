"""Core redaction logic for document images."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


from redact_id.main import FULL_BLUR_CLASSES, PARTIAL_BLUR_CLASSES
from redact_id.settings import settings


# ---------- Data model ----------
@dataclass(frozen=True)
class Detection:
    """Represents a detected object in an image."""
    cls_id: int
    conf: float
    xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def resize_if_needed(
    img: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, float]:
    """
    Resize image proportionally if larger than max_dim.

    Returns:
        resized_img, scale_factor
    """
    h, w = img.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)

    if scale >= 1.0:
        return img, 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


# ---------- Utilities ----------
def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """Clip bounding box to image boundaries."""
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _auto_kernel_from_box(box_w: int, box_h: int) -> Tuple[int, int]:
    """Auto-select blur kernel size based on box dimensions."""
    base=(min(box_w, box_h) // 5) * 2 + 1
    k = int(max(21, min(81,base*settings.BLUR_STRENGTH )))
    if k % 2 == 0:
        k += 1
    return (k, k)


def _blur_region(img: np.ndarray, box: Tuple[int, int, int, int], kernel: Optional[Tuple[int, int]] = None) -> None:
    """Apply Gaussian blur to a region of the image (in-place)."""
    h, w = img.shape[:2]
    clipped = _clip_box(*box, w=w, h=h)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = kernel or _auto_kernel_from_box(x2 - x1, y2 - y1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, k, 0)


def _partial_blur_keep_right(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    keep_ratio: float = settings.KEEP_RATIO,
    kernel: Optional[Tuple[int, int]] = None,
) -> None:
    """Blur left portion of box, keep right portion visible (in-place)."""
    keep_ratio = float(np.clip(keep_ratio, 0.05, 0.95))
    h, w = img.shape[:2]
    clipped = _clip_box(*box, w=w, h=h)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    if bw <= 2:
        return

    blur_x2 = int(x2 - bw * keep_ratio)
    blur_x2 = max(x1 + 1, min(blur_x2, x2 - 1))

    roi = img[y1:y2, x1:blur_x2]
    if roi.size == 0:
        return
    k = kernel or _auto_kernel_from_box(blur_x2 - x1, y2 - y1)
    img[y1:y2, x1:blur_x2] = cv2.GaussianBlur(roi, k, 0)


# ---------- Core logic ----------
def redact_image_bgr(
    img_bgr: np.ndarray,
    detections: List[Detection],
    keep_ratio: float = settings.KEEP_RATIO,
    partial_mode: str = "best",
    full_kernel: Optional[Tuple[int, int]] = settings.BLUR_KERNEL,
    partial_kernel: Optional[Tuple[int, int]] = settings.BLUR_KERNEL,
) -> np.ndarray:
    """
    Apply redaction rules to an image (BGR). Returns redacted image.
    """
    if partial_mode not in {"best", "all"}:
        raise ValueError("partial_mode must be 'best' or 'all'")

    out = img_bgr.copy()

    # Always full-blur mandatory classes
    for det in detections:
        if det.cls_id in FULL_BLUR_CLASSES:
            _blur_region(out, det.xyxy, kernel=full_kernel)

    # Collect partial-group detections
    partial_candidates = [d for d in detections if d.cls_id in PARTIAL_BLUR_CLASSES]

    if not partial_candidates:
        return out

    # Decide which boxes get partial blur
    best_key = None
    if partial_mode == "best":
        best = max(partial_candidates, key=lambda d: d.conf)
        # value-based identity (stable)
        best_key = (best.cls_id, best.conf, best.xyxy)

    # Apply partial vs full blur
    for det in partial_candidates:
        is_best = (
            partial_mode == "all"
            or (best_key is not None and (det.cls_id, det.conf, det.xyxy) == best_key)
        )

        if is_best:
            _partial_blur_keep_right(
                out,
                det.xyxy,
                keep_ratio=keep_ratio,
                kernel=partial_kernel,
            )
        else:
            _blur_region(out, det.xyxy, kernel=full_kernel)

    return out



def detections_from_ultralytics_result(result) -> List[Detection]:
    """Convert Ultralytics YOLO Result to Detection list."""
    dets: List[Detection] = []
    if result is None or getattr(result, "boxes", None) is None:
        return dets

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        dets.append(
            Detection(
                cls_id=int(c),
                conf=float(p),
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
            )
        )
    return dets


def get_blur_type(
    det: Detection,
    detections: List[Detection],
    partial_mode: str = "best",
) -> str:
    """
    Determine blur type for a detection.

    Args:
        det: Detection to classify
        detections: Full list of detections
        partial_mode: "best" or "all"

    Returns:
        "full" or "partial"
    """
    if det.cls_id in FULL_BLUR_CLASSES:
        return "full"

    if det.cls_id not in PARTIAL_BLUR_CLASSES:
        return "unknown"

    if partial_mode == "all":
        return "partial"

    # partial_mode == "best": only highest conf gets partial blur
    partial_candidates = [d for d in detections if d.cls_id in PARTIAL_BLUR_CLASSES]
    if not partial_candidates:
        return "full"

    best = max(partial_candidates, key=lambda d: d.conf)
    # Use stable comparison by value, not identity
    best_key = (best.cls_id, best.conf, best.xyxy)
    det_key = (det.cls_id, det.conf, det.xyxy)

    return "partial" if det_key == best_key else "full"


# ---------- Encoding helper ----------
def encode_image_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    """
    Encode image as JPEG bytes.

    Args:
        img: Image in BGR format
        quality: JPEG quality (1-100)

    Returns:
        JPEG bytes
    """
    success, encoded = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not success:
        raise RuntimeError("Failed to encode image as JPEG")
    return encoded.tobytes()


# ---------- Async API ----------
async def detect_async(
    image_bytes: bytes,
    model: YOLO,
    conf_thres: float = 0.25,
) -> tuple[np.ndarray, List[Detection], tuple[int, int], float]:
    """
    Async YOLO detection on image.

    Args:
        image_bytes: Input image as bytes
        model: Loaded YOLO model
        conf_thres: Confidence threshold

    Returns:
        Tuple of (decoded_image, detections)
    """

    def _detect():
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")

        original_h, original_w = img.shape[:2]

        img, scale = resize_if_needed(img, settings.MAX_IMAGE_DIMENSION)

        results = model.predict(source=img, conf=conf_thres, verbose=False)
        dets = detections_from_ultralytics_result(results[0])

        return img, dets, (original_w, original_h), scale

    return await asyncio.to_thread(_detect)


async def redact_and_encode_async(
    img: np.ndarray,
    detections: List[Detection],
    keep_ratio: float = settings.KEEP_RATIO,
    partial_mode: str = "best",
    jpeg_quality: int = 90,
) -> bytes:
    """
    Async redaction and encoding.

    Args:
        img: Decoded image in BGR format
        detections: List of detections
        keep_ratio: Ratio of right portion to keep visible
        partial_mode: "best" or "all"
        jpeg_quality: JPEG quality (1-100)

    Returns:
        Redacted image as JPEG bytes
    """
    def _redact_and_encode():
        # Apply redaction
        redacted = redact_image_bgr(img, detections, keep_ratio=keep_ratio, partial_mode=partial_mode)

        # Encode to JPEG
        return encode_image_jpeg(redacted, jpeg_quality)

    return await asyncio.to_thread(_redact_and_encode)


