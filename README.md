![Docker Pulls](https://img.shields.io/docker/pulls/aminook/redact-id)
# RedactID
Production-ready document redaction API for sensitive IDs and personal data

<img src="assets/sample.png" width="400" height="235">
<img src="assets/sample_json.png" width="400" height="235">

RedactID is a **FastAPI + YOLO** service that detects and redacts sensitive information (IDs, passport numbers, addresses) from document images — built for **real compliance workflows**, not toy demos.

> Clean MVP with production habits: async execution, configurable redaction policy, and audit-friendly outputs.

---

## Why RedactID

Most redaction tools are:
- manual and slow
- opaque (hard to verify what happened)
- painful to integrate or audit

**RedactID fixes this by design:**
- privacy-first redaction logic
- fast async inference
- configurable partial vs full blur
- audit-ready JSON metadata
- drop-in API for existing systems

---

## Key Capabilities

- YOLO-based PII field detection (IDs, passports, addresses, etc.)
- Redaction policy engine:
  - **Full blur** for highly sensitive fields
  - **Partial blur** (keep last digits visible) when allowed
- Two endpoints:
  - `POST /redact` → returns redacted JPEG bytes
  - `POST /redact/json` → returns redacted image (base64) + metadata
- Single-pass inference (YOLO runs once per request)
- Production safeguards:
  - file size limits
  - image dimension limits + auto-resize
  - structured logging + request id headers
- Async, non-blocking design (`asyncio.to_thread` CPU offload)

---

## Redaction Logic

### Example Supported Classes

| Field | Policy |
|------|--------|
| Address | Full blur |
| Long Passport Number | Full blur |
| ID Number | Partial or full |
| NHI ID | Partial or full |
| Passport Number | Partial or full |

### Partial Blur Strategy

Partial blur keeps the **rightmost portion visible** (e.g., last 3–4 digits):

XXXXXXXXXX6789

This is useful for:
- human verification
- compliance review
- support workflows (confirming the right document without exposing full IDs)

### Partial Modes

- `best` (default)  
  Only the **highest-confidence** ID-like field keeps partial visibility; other ID-like fields are fully blurred.  
  Recommended for typical single-document images.

- `all`  
  All ID-like fields get partial blur.  
  Useful when multiple documents appear in one image.

---

## API Overview

### 1) Redact Image (binary output)

```bash
curl -X POST "http://localhost:8000/redact?partial_mode=best&keep_ratio=0.3" \
  -F "file=@document.jpg" \
  --output redacted.jpg
```

**Response headers include:**
- `X-Request-ID`
- `X-Processing-Time-Ms`
- `X-Detections-Count`
- `X-Partial-Mode`

### 2) Redact with Metadata (JSON)

```bash
curl -X POST "http://localhost:8000/redact/json" \
  -F "file=@document.jpg"
```

**Returns:**
- base64-encoded redacted image
- detected fields + confidence
- blur type per detection (full / partial)
- original vs processed size + scale factor
- request id + latency

**Example JSON output (trimmed):**

```json
{
  "request_id": "a3b7c9d2",
  "detection_count": 2,
  "processing_time_ms": 214.8,
  "detections": [
    {
      "class_name": "ID Number",
      "confidence": 0.95,
      "blur_type": "partial"
    },
    {
      "class_name": "Address",
      "confidence": 0.88,
      "blur_type": "full"
    }
  ]
}
```

---

## Quick Start (Local)

```bash
# Install dependencies
uv sync

# Set model path (your trained YOLO weights)
export MODEL_PATH=path/to/best.pt

# Run server
uv run python -m src.redact_id.main
```

**Docs:** OpenAPI UI at http://localhost:8000/docs

---

## Run with Docker (Docker Hub)

**Image:** `aminook/redact-id:latest`

**Recommended:** copy `.env.docker.example` → `.env.docker` and edit values.

```bash
docker run --rm \
--env-file .env.docker \
-p 8000:8000 \
-e MODEL_PATH=/model/trained_model.pt \
-e REDACTION_POLICY_PATH=/policy/redaction_policy.json \
-v $(pwd)/model:/model \
-v $(pwd)/policy:/policy \
aminook/redact-id:latest
```

**Notes:**
- The host folder `./model` must contain `trained_model.pt`
- The host folder `./policy` must contain `redaction_policy.json`
- Paths inside the container must match MODEL_PATH and REDACTION_POLICY_PATH
---

## Performance (Typical)

For a ~1024×768 image (CPU inference; varies by hardware/model):

- YOLO inference: ~150–200 ms
- Redaction + encoding: ~30–40 ms
- **Total: ~200–250 ms**

**Optimizations used:**
- one inference pass per request
- async CPU offloading
- optional auto-resize for overly large images

---

## Configuration

Configuration is environment-driven via `pydantic-settings`.

**Common env vars:**
- `MODEL_PATH` (required): path to YOLO weights
- `MAX_FILE_SIZE`: max upload size in bytes
- `MAX_IMAGE_DIMENSION`: resize guardrail
- `KEEP_RATIO`: how much of the right side stays visible (partial blur)
- `BLUR_STRENGTH`: scales blur kernel selection
- `REDACTION_POLICY_PATH` (optional): JSON policy override

See `.env.docker.example` for a working template.

---

## What This Project Demonstrates

- production-grade FastAPI app factory + lifespan model loading
- async execution with CPU offload (no blocking endpoints)
- practical redaction rules (not "blur the whole image")
- stable, audit-friendly outputs (JSON metadata + headers)
- configurable policy design suitable for regulated pipelines

---

## Project Status

MVP complete — easy to extend:

- OCR verification (optional) and post-processing rules
- PDF input support
- per-customer / per-jurisdiction policy bundles
- auth + rate limiting for production deployments

---

## License

MIT (or client-specific licensing)