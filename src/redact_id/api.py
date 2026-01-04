"""FastAPI application for document redaction service."""

import base64
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from ultralytics import YOLO

from .core import (
    detect_async,
    redact_and_encode_async,
    get_blur_type
)
from .main import CLASS_NAMES
from .settings import settings

logger = logging.getLogger("redact_id")

# Response models
class DetectionInfo(BaseModel):
    """Information about a single detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    blur_type: str  # "full" or "partial"


class RedactJSONResponse(BaseModel):
    """JSON response with redacted image and metadata."""
    request_id: str
    filename: str
    detections: List[DetectionInfo]
    detection_count: int
    processing_time_ms: float
    redacted_image_base64: str
    original_image_size: Dict[str, int]
    processed_image_size: Dict[str, int]
    scale_factor: float



# Global model instance
_model: Optional[YOLO] = None

def get_model_path() -> str:
    model_path = os.getenv("YOLO_MODEL_PATH", settings.MODEL_PATH)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    model_path = app.state.model_path
    logger.info(f"Loading YOLO model from {model_path}")
    _model = YOLO(model_path)
    logger.info("Model loaded successfully")
    yield
    _model = None
    logger.info("Model unloaded")


def create_app() -> FastAPI:
    """
    App factory. Reads env at runtime, not import time.
    """
    app = FastAPI(
        title="RedactID API",
        description="Document redaction service using YOLO object detection",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.state.model_path = get_model_path()

    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "service": "RedactID", "version": "0.2.0"}

    @app.post("/redact")
    async def redact_document(
        file: UploadFile = File(..., description="Image file to redact"),
        conf_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
        keep_ratio: float = Query(0.30, ge=0.05, le=0.95, description="Ratio to keep visible"),
        partial_mode: str = Query("best", pattern="^(best|all)$", description="Partial blur mode: best or all"),
        jpeg_quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
    ) -> Response:
        """
        Redact sensitive information from document image.

        Returns redacted image as JPEG.
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            # Read and validate file size
            image_bytes = await file.read()
            file_size = len(image_bytes)

            if file_size == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            if file_size > settings.MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File too large (max {settings.MAX_FILE_SIZE // 1024 // 1024}MB)")

            # Validate image dimensions
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")

            height, width = img.shape[:2]
            img, detections, _, _ = await detect_async(image_bytes, _model, conf_threshold)
            # Redact and encode
            redacted_bytes = await redact_and_encode_async(
                img,
                detections,
                keep_ratio=keep_ratio,
                partial_mode=partial_mode,
                jpeg_quality=jpeg_quality,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Log request
            logger.info(
                f"req_id={request_id} filename={file.filename} "
                f"size={width}x{height} file_size={file_size} "
                f"detections={len(detections)} latency={elapsed_ms:.2f}ms"
            )

            return Response(
                content=redacted_bytes,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'inline; filename="redacted_{file.filename}"',
                    "X-Request-ID": request_id,
                    "X-Processing-Time-Ms": f"{elapsed_ms:.2f}",
                    "X-Detections-Count": str(len(detections)),
                    "X-Partial-Mode": partial_mode,
                },
            )

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"req_id={request_id} error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    @app.post("/redact/json", response_model=RedactJSONResponse)
    async def redact_document_json(
        file: UploadFile = File(..., description="Image file to redact"),
        conf_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
        keep_ratio: float = Query(0.30, ge=0.05, le=0.95, description="Ratio to keep visible"),
        partial_mode: str = Query("best", pattern="^(best|all)$", description="Partial blur mode: best or all"),
        jpeg_quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
    ) -> RedactJSONResponse:
        """
        Redact sensitive information and return JSON with metadata.

        Returns:
            JSON with base64-encoded redacted image, detections, and metadata.
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            # Read and validate
            image_bytes = await file.read()
            file_size = len(image_bytes)

            if file_size == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            if file_size > settings.MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File too large")

            # Run YOLO detection ONCE
            img, detections, (orig_w, orig_h), scale = await detect_async(
                image_bytes, _model, conf_threshold
            )

            h, w = img.shape[:2]

            height, width = img.shape[:2]

            # Determine blur types using stable comparison
            detection_infos = []
            for det in detections:
                blur_type = get_blur_type(det, detections, partial_mode)

                detection_infos.append(
                    DetectionInfo(
                        class_id=det.cls_id,
                        class_name=CLASS_NAMES.get(det.cls_id, f"Class_{det.cls_id}"),
                        confidence=det.conf,
                        bbox=list(det.xyxy),
                        blur_type=blur_type,
                    )
                )

            # Redact and encode (using already-decoded img and detections)
            redacted_bytes = await redact_and_encode_async(
                img,
                detections,
                keep_ratio=keep_ratio,
                partial_mode=partial_mode,
                jpeg_quality=jpeg_quality,
            )

            # Encode to base64
            redacted_b64 = base64.b64encode(redacted_bytes).decode("utf-8")

            elapsed_ms = (time.time() - start_time) * 1000

            # Log
            logger.info(
                f"req_id={request_id} filename={file.filename} "
                f"size={width}x{height} detections={len(detections)} "
                f"latency={elapsed_ms:.2f}ms endpoint=json"
            )

            return RedactJSONResponse(
                request_id=request_id,
                filename=file.filename or "unknown",
                original_image_size={"width": orig_w, "height": orig_h},
                processed_image_size={"width": w, "height": h},
                scale_factor=round(scale, 4),
                detections=detection_infos,
                detection_count=len(detections),
                processing_time_ms=round(elapsed_ms, 2),
                redacted_image_base64=redacted_b64,
            )

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"req_id={request_id} error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return app
