"""
Inference Engine for VGG16 Fire & Smoke Detection
"""

import time
from typing import Dict, Any, Union
import numpy as np
from PIL import Image
import config
from utils.preprocessing import preprocess_for_vgg16
from models.model_loader import ModelStatus

def run_vgg16_inference(
    image_input: Union[Image.Image, np.ndarray, str],
    model_status: ModelStatus,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Executes deep learning inference on an input image.
    Strictly uses the trained VGG16 model. If weights are missing,
    it returns an informative error without fabricating fake detections.
    """
    if not model_status.is_loaded or model_status.model is None:
        return {
            "success": False,
            "error": "MODEL_NOT_LOADED",
            "message": (
                "VGG16 Model weights are not available. "
                "Inference cannot be performed without trained weights. "
                "Please train the model via 'python models/train_model.py' or check Model Information."
            )
        }

    try:
        # Preprocessing step
        start_time = time.perf_counter()
        preprocessed = preprocess_for_vgg16(image_input)
        
        # Inference step
        preds = model_status.model.predict(preprocessed, verbose=0)
        latency_ms = (time.perf_counter() - start_time) * 1000

        classes = model_status.classes
        output_dim = preds.shape[-1]
        probs: Dict[str, float] = {}

        if output_dim == 1:
            # Binary sigmoid output (e.g., 0: Fire, 1: Normal or vice versa)
            prob_val = float(preds[0][0])
            # Default convention: higher value = class 1, lower value = class 0
            if len(classes) >= 2:
                probs[classes[0]] = 1.0 - prob_val
                probs[classes[1]] = prob_val
                best_class_idx = 1 if prob_val >= 0.5 else 0
                predicted_class = classes[best_class_idx]
                confidence = float(probs[predicted_class])
            else:
                predicted_class = "Fire" if prob_val >= 0.5 else "Normal"
                confidence = prob_val if prob_val >= 0.5 else (1.0 - prob_val)
                probs = {"Fire": prob_val, "Normal": 1.0 - prob_val}
        else:
            # Multi-class Softmax outputs
            raw_probs = preds[0]
            # Normalize just in case
            exp_probs = raw_probs / (np.sum(raw_probs) + 1e-9)
            best_class_idx = int(np.argmax(exp_probs))
            
            for idx, cname in enumerate(classes):
                probs[cname] = float(exp_probs[idx]) if idx < len(exp_probs) else 0.0
                
            predicted_class = classes[best_class_idx]
            confidence = float(exp_probs[best_class_idx])

        # Flag safety threshold
        is_above_threshold = confidence >= confidence_threshold

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "all_probabilities": {k: round(v, 4) for k, v in probs.items()},
            "latency_ms": round(latency_ms, 2),
            "is_above_threshold": is_above_threshold,
            "supported_classes": classes
        }

    except Exception as e:
        return {
            "success": False,
            "error": "INFERENCE_FAILED",
            "message": f"Inference execution failed: {str(e)}"
        }
