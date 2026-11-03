"""
Input Validation Utilities for FIREGUARD AI
"""

from typing import Tuple, Optional
from PIL import Image, UnidentifiedImageError
import io

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_MB = 25

def validate_uploaded_image(uploaded_file) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validates uploaded file against allowed formats, size limits, and image corruption.
    Returns:
        (is_valid, error_message, PIL_image_instance)
    """
    if uploaded_file is None:
        return False, "No file uploaded.", None

    # Check extension
    filename = uploaded_file.name.lower()
    ext = filename.split(".")[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type (.{ext}). Please upload a JPG, JPEG, or PNG image.", None

    # Check file size
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File exceeds maximum size limit of {MAX_FILE_SIZE_MB}MB (Current: {size_mb:.2f}MB).", None

    # Check image readability and corruption
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verify integrity
        # Re-open because verify() closes the stream or leaves it unusable
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return True, None, image
    except (UnidentifiedImageError, OSError, Exception) as e:
        return False, f"Corrupted or invalid image file: {str(e)}", None


def validate_image_array(img_array) -> Tuple[bool, Optional[str]]:
    """Validates numpy image array dimensions and channels."""
    if img_array is None:
        return False, "Image array is null."
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return False, f"Invalid image dimensions: {img_array.shape}. Expected (H, W, 3)."
    return True, None
