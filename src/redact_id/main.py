# src/redact_id/main.py
import json
import logging
import os
import uvicorn
from redact_id.consts import DEFAULT_CLASS_NAMES, DEFAULT_FULL_BLUR_CLASSES, DEFAULT_PARTIAL_BLUR_CLASSES
from redact_id.settings import settings



def load_redaction_policy(path: str | None):
    if not path:
        return (
            DEFAULT_CLASS_NAMES,
            DEFAULT_FULL_BLUR_CLASSES,
            DEFAULT_PARTIAL_BLUR_CLASSES,
        )

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load redaction policy: {e}")

    if "class_names" not in data:
        raise ValueError("redaction policy must define 'class_names'")

    class_names = {int(k): v for k, v in data["class_names"].items()}

    full_blur = set(map(int, data.get("full_blur_classes", [])))
    partial_group = set(map(int, data.get("partial_group_classes", [])))

    # Safety checks
    unknown = (full_blur | partial_group) - set(class_names.keys())
    if unknown:
        raise ValueError(f"Unknown class IDs in policy: {unknown}")

    overlap = full_blur & partial_group
    if overlap:
        raise ValueError(
            f"Classes cannot be both full and partial blur: {overlap}"
        )

    return class_names, full_blur, partial_group

CLASS_NAMES, FULL_BLUR_CLASSES, PARTIAL_BLUR_CLASSES = load_redaction_policy(
    settings.REDACTION_POLICY_PATH
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"{name} must be an integer (got {raw!r})") from e




def main() -> None:
    host = settings.HOST
    port = settings.PORT
    reload = settings.RELOAD


    uvicorn.run(
        "redact_id.api:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()
