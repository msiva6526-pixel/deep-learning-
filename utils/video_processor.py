"""
CCTV Video Processing and Frame-by-Frame Inference Engine
"""

import cv2
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from PIL import Image
import numpy as np

import config
from models.model_loader import ModelStatus
from models.predict import run_vgg16_inference
from utils.preprocessing import cv2_frame_to_pil

def analyze_cctv_video(
    video_path: str,
    model_status: ModelStatus,
    target_fps_sample: float = 2.0,
    conf_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Performs frame-by-frame deep learning surveillance inference on an uploaded CCTV video.
    
    Args:
        video_path: Local path to the MP4/AVI video file
        model_status: Loaded VGG16 ModelStatus object
        target_fps_sample: How many frames per second of video to sample (e.g., 2.0 = analyze a frame every 0.5s)
        conf_threshold: Alert threshold
        progress_callback: Optional function (progress: float, status_text: str)
    
    Returns:
        Dictionary containing video metrics, detection timeline, keyframes, and hazard alerts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "success": False,
            "error": "CANNOT_OPEN_VIDEO",
            "message": f"Could not decode video file: {video_path}"
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / video_fps if total_frames > 0 else 0.0

    # Determine frame step (how many frames to skip between inferences)
    step = max(1, int(round(video_fps / target_fps_sample)))

    timeline: List[Dict[str, Any]] = []
    keyframes: List[Dict[str, Any]] = []

    frame_idx = 0
    analyzed_count = 0
    fire_count = 0
    smoke_count = 0
    normal_count = 0
    peak_fire_conf = 0.0
    first_fire_sec = None

    start_proc_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample at calculated interval
        if frame_idx % step == 0:
            current_sec = round(frame_idx / video_fps, 2)
            
            # Convert to PIL Image for VGG16 inference
            pil_frame = cv2_frame_to_pil(frame)
            
            # Run VGG16 prediction
            res = run_vgg16_inference(pil_frame, model_status, confidence_threshold=conf_threshold)
            
            if res["success"]:
                pclass = res["predicted_class"]
                conf = res["confidence"]
                conf_pct = res["confidence_percentage"]
                probs = res["all_probabilities"]
                fire_prob = probs.get("Fire", 0.0)

                class_lower = pclass.lower()
                is_fire = "fire" in class_lower
                is_smoke = "smoke" in class_lower

                if is_fire:
                    fire_count += 1
                    if first_fire_sec is None:
                        first_fire_sec = current_sec
                    if conf > peak_fire_conf:
                        peak_fire_conf = conf
                elif is_smoke:
                    smoke_count += 1
                else:
                    normal_count += 1

                # Record timeline entry
                timeline.append({
                    "second": current_sec,
                    "frame_index": frame_idx,
                    "predicted_class": pclass,
                    "confidence": conf,
                    "confidence_pct": conf_pct,
                    "fire_probability": round(fire_prob * 100, 1),
                    "is_alert": is_fire or is_smoke
                })

                # Annotate keyframe for visualization
                annotated = frame.copy()
                h, w = frame.shape[:2]
                
                # Draw HUD border and banner
                if is_fire:
                    border_color = (0, 0, 239)   # Red
                    banner_text = f"CRITICAL FIRE ALERT: {conf_pct:.1f}%"
                elif is_smoke:
                    border_color = (0, 158, 245) # Amber
                    banner_text = f"SMOKE WARNING: {conf_pct:.1f}%"
                else:
                    border_color = (0, 200, 50)  # Green
                    banner_text = f"NORMAL SECURE: {conf_pct:.1f}%"

                # Draw top alert banner
                cv2.rectangle(annotated, (0, 0), (w, 45), (15, 23, 42), -1)
                cv2.rectangle(annotated, (0, 0), (w, h), border_color, 4)
                cv2.putText(annotated, banner_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, border_color, 2, cv2.LINE_AA)
                cv2.putText(annotated, f"T+{current_sec:.1f}s", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

                # Store keyframe (all alert frames or every 5th regular frame)
                if is_fire or is_smoke or len(keyframes) < 4 or (analyzed_count % 5 == 0 and len(keyframes) < 12):
                    keyframes.append({
                        "second": current_sec,
                        "class": pclass,
                        "confidence": conf_pct,
                        "image": cv2_frame_to_pil(annotated)
                    })

                analyzed_count += 1

            # Update progress callback if given
            if progress_callback and total_frames > 0:
                prog_val = min(1.0, frame_idx / total_frames)
                progress_callback(prog_val, f"Analyzing frame {frame_idx}/{total_frames} (T+{current_sec:.1f}s)...")

        frame_idx += 1

    cap.release()
    total_latency_sec = time.perf_counter() - start_proc_time

    return {
        "success": True,
        "filename": Path(video_path).name,
        "total_frames": total_frames,
        "video_fps": round(video_fps, 2),
        "duration_sec": round(duration_sec, 2),
        "analyzed_frames": analyzed_count,
        "fire_frames": fire_count,
        "smoke_frames": smoke_count,
        "normal_frames": normal_count,
        "has_fire": fire_count > 0,
        "has_smoke": smoke_count > 0,
        "first_fire_sec": first_fire_sec,
        "peak_fire_confidence": round(peak_fire_conf * 100, 2),
        "total_processing_time_sec": round(total_latency_sec, 2),
        "processing_fps": round(analyzed_count / total_latency_sec, 1) if total_latency_sec > 0 else 0,
        "timeline": timeline,
        "keyframes": keyframes
    }
