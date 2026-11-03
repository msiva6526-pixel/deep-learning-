"""
Image Preprocessing Utilities for VGG16 Deep Learning Pipeline
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Union
import config

def preprocess_for_vgg16(
    image_input: Union[Image.Image, np.ndarray, str],
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    apply_mean_subtraction: bool = False
) -> np.ndarray:
    """
    Prepares an image for VGG16 model inference.
    Steps:
      1. Load / Convert to RGB numpy array.
      2. Resize to (224, 224).
      3. Expand batch dimension to (1, 224, 224, 3).
      4. Apply standard VGG16 preprocessing (RGB -> BGR, mean subtraction: [103.939, 116.779, 123.68]).
    
    Returns:
        np.ndarray with shape (1, 224, 224, 3) ready for model.predict()
    """
    # Convert PIL Image or path to RGB numpy array
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
        img_arr = np.array(pil_img)
    elif isinstance(image_input, Image.Image):
        img_arr = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img_arr = image_input.copy()
        if len(img_arr.shape) == 2:  # Grayscale
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:  # RGBA
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize to target size (224, 224) using high-quality INTER_AREA / INTER_LINEAR
    resized_img = cv2.resize(img_arr, target_size, interpolation=cv2.INTER_AREA)

    # Cast to float32
    x = resized_img.astype(np.float32)

    # Add batch dimension -> (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)

    # Only apply manual mean subtraction if explicitly requested (otherwise model's built-in preprocess layer handles it)
    if apply_mean_subtraction:
        x = x[..., ::-1]
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def cv2_frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR frame into a PIL RGB Image."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def pil_to_cv2_frame(image: Image.Image) -> np.ndarray:
    """Converts a PIL RGB Image into an OpenCV BGR frame."""
    rgb_array = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
