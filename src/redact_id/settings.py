# src/redact_id/settings.py
import sys
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from redact_id.consts import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_RELOAD, DEFAULT_MAX_FILE_SIZE, \
    DEFAULT_MAX_IMAGE_DIMENSION, DEFAULT_KEEP_RATIO, DEFAULT_BLUR_KERNEL, DEFAULT_BLUR_STRENGTH


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_parse_none_str='None'
    )
    # === Model ===
    MODEL_PATH: Path = Field(default=Path('xxx'), description="Path to YOLO model weights")

    REDACTION_POLICY_PATH: Optional[Path] = Field(
        default=None,
        description="Optional path to redaction policy JSON; if omitted, defaults are used",
    )

    # === Server ===
    HOST: str = Field(default=DEFAULT_HOST)
    PORT: int = Field(default=DEFAULT_PORT, ge=1, le=65535)
    RELOAD: bool = Field(default=DEFAULT_RELOAD)
    MAX_FILE_SIZE: int = Field(default=DEFAULT_MAX_FILE_SIZE)
    MAX_IMAGE_DIMENSION: int = Field(default=DEFAULT_MAX_IMAGE_DIMENSION)
    KEEP_RATIO: float = Field(default=DEFAULT_KEEP_RATIO)
    BLUR_KERNEL: Optional[int] = Field(default=DEFAULT_BLUR_KERNEL)
    BLUR_STRENGTH: int = Field(default=DEFAULT_BLUR_STRENGTH)

    @field_validator("MODEL_PATH")
    @classmethod
    def model_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"MODEL_PATH does not exist: {v}")
        return v

    @field_validator("REDACTION_POLICY_PATH")
    @classmethod
    def policy_must_exist_if_set(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"REDACTION_POLICY_PATH does not exist: {v}")
        return v


def load_settings_or_die() -> Settings:
    try:
        s = Settings()
    except ValidationError as e:
        # One clean message, no scary traceback
        print("[CONFIG ERROR] Invalid environment configuration:", file=sys.stderr)
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "invalid value")
            print(f"  - {loc}: {msg}", file=sys.stderr)
        sys.exit(2)

    return s

settings = load_settings_or_die()









