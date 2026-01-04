
DEFAULT_HOST='0.0.0.0'
DEFAULT_PORT=8000
DEFAULT_RELOAD=1

# Security limits
DEFAULT_MAX_FILE_SIZE = 5242880   # 5MB
DEFAULT_MAX_IMAGE_DIMENSION = 1024  # pixels


# Class names mapping
DEFAULT_CLASS_NAMES = {
    0: "Address",
    1: "Birth Date",
    2: "ID Number",
    3: "Long Passport Number",
    4: "NHI ID",
    5: "Name",
    6: "Others",
    7: "Passport Number",
    8: "Passport Number Vertical",
}

# ---------- Config ----------
DEFAULT_FULL_BLUR_CLASSES = {0,3,6,8}
DEFAULT_PARTIAL_BLUR_CLASSES = {2, 4, 7}

DEFAULT_KEEP_RATIO = 0.25  # Keep right 30% visible for partial blur
DEFAULT_BLUR_KERNEL = None  # Auto-pick from box size
DEFAULT_BLUR_STRENGTH = 5



