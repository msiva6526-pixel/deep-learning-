"""
VGG16 Deep Learning Model Loader & Cache Manager
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import streamlit as st

# Configure TensorFlow logs before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
from keras.applications.vgg16 import VGG16
import config

class ModelStatus:
    def __init__(
        self,
        is_loaded: bool,
        model: Optional[Any],
        classes: List[str],
        weights_path: Optional[str],
        status_message: str,
        architecture: str = "VGG16 Transfer Learning"
    ):
        self.is_loaded = is_loaded
        self.model = model
        self.classes = classes
        self.weights_path = weights_path
        self.status_message = status_message
        self.architecture = architecture


@st.cache_resource(show_spinner=False)
def load_fire_model(custom_path: Optional[str] = None) -> ModelStatus:
    """
    Loads and caches the fine-tuned VGG16 model.
    Checks:
      1. Explicit custom path if provided.
      2. DEFAULT_MODEL_PATH (.keras).
      3. FALLBACK_H5_PATH (.h5).
    If no trained weights exist, gracefully informs the user without faking results.
    """
    target_path = None
    if custom_path and Path(custom_path).exists():
        target_path = Path(custom_path)
    elif config.DEFAULT_MODEL_PATH.exists():
        target_path = config.DEFAULT_MODEL_PATH
    elif config.FALLBACK_H5_PATH.exists():
        target_path = config.FALLBACK_H5_PATH

    # Check for class mappings
    classes = config.DEFAULT_CLASSES
    if config.CLASS_INDICES_PATH.exists():
        try:
            with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                class_map = json.load(f)
                # Map may be {"0": "Fire", "1": "Normal"} or {"Fire": 0, "Normal": 1}
                if all(isinstance(k, str) and k.isdigit() for k in class_map.keys()):
                    classes = [class_map[str(i)] for i in range(len(class_map))]
                else:
                    sorted_classes = sorted(class_map.items(), key=lambda item: item[1])
                    classes = [item[0] for item in sorted_classes]
        except Exception:
            classes = config.DEFAULT_CLASSES

    if target_path is None:
        return ModelStatus(
            is_loaded=False,
            model=None,
            classes=classes,
            weights_path=None,
            status_message=(
                "No trained VGG16 weights found at 'models/saved_models/vgg16_fire_model.keras'. "
                "Please run 'python models/train_model.py' to train on your Kaggle FIRE dataset, "
                "or place trained weights in the saved_models folder."
            )
        )

    try:
        # Load the custom trained Keras VGG16 model with registered preprocess function
        custom_objects = {
            "preprocess_input": keras.applications.vgg16.preprocess_input
        }
        model = keras.models.load_model(str(target_path), custom_objects=custom_objects)
        
        # Verify model output shape matches classes length
        output_dim = model.output_shape[-1]
        if output_dim == 1 and len(classes) != 2:
            classes = ["Fire", "Normal"]
        elif output_dim > 1 and len(classes) != output_dim:
            if output_dim == 3:
                classes = config.EXTENDED_CLASSES
            elif output_dim == 2:
                classes = config.DEFAULT_CLASSES
            else:
                classes = [f"Class_{i}" for i in range(output_dim)]

        return ModelStatus(
            is_loaded=True,
            model=model,
            classes=classes,
            weights_path=str(target_path),
            status_message=f"Model successfully loaded from '{target_path.name}' ({len(classes)} classes)."
        )
    except Exception as e:
        return ModelStatus(
            is_loaded=False,
            model=None,
            classes=classes,
            weights_path=str(target_path),
            status_message=f"Error loading model weights from '{target_path}': {str(e)}"
        )


def get_model_summary_info(model) -> Dict[str, Any]:
    """Extracts architectural metadata for dashboard & model info page."""
    if model is None:
        return {
            "total_layers": 0,
            "total_params": 0,
            "trainable_params": 0,
            "input_shape": list(config.INPUT_SHAPE)
        }
    
    total_params = model.count_params()
    trainable_params = sum(keras.ops.size(w) for w in model.trainable_weights) if hasattr(model, "trainable_weights") else 0
    total_layers = len(model.layers) if hasattr(model, "layers") else 0

    return {
        "total_layers": total_layers,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "input_shape": list(config.INPUT_SHAPE)
    }
