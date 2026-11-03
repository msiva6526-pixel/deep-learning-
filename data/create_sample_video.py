"""
Generates a realistic 6-second CCTV surveillance video (30 FPS) for testing video inference:
  - 0s to 3s: Normal server room monitoring
  - 3s to 6s: Fire incident flare-up and smoke dispersion
"""

import cv2
import numpy as np
from pathlib import Path

sample_video_dir = Path(__file__).resolve().parent / "sample_videos"
sample_video_dir.mkdir(parents=True, exist_ok=True)
output_video_path = sample_video_dir / "cctv_surveillance_demo.mp4"

width, height = 640, 480
fps = 25
duration_sec = 6
total_frames = fps * duration_sec

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

for f in range(total_frames):
    # Base background: Server Room
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = [30, 25, 20]  # dark blue-gray

    # Draw server racks
    for x in [60, 200, 340, 480]:
        cv2.rectangle(frame, (x, 100), (x + 100, 420), (45, 40, 35), -1)
        cv2.rectangle(frame, (x, 100), (x + 100, 420), (70, 65, 55), 2)
        # Server LEDs
        for y in range(120, 400, 25):
            cv2.circle(frame, (x + 20, y), 3, (255, 180, 0), -1)
            cv2.circle(frame, (x + 40, y), 3, (0, 255, 100), -1)

    # Ceiling light
    cv2.rectangle(frame, (150, 20), (490, 40), (220, 230, 240), -1)

    # Time calculation
    sec = f / fps
    time_str = f"2026-10-29 01:05:{int(sec):02d}.{int((sec % 1) * 100):02d}"

    # Fire Flare-up starting after 3 seconds
    if sec >= 2.8:
        progress = min(1.0, (sec - 2.8) / 1.5)
        flame_h = int(120 * progress)
        
        # Flame glow
        cv2.circle(frame, (280, 350), int(100 * progress), (0, 60, int(180 * progress)), -1)

        # Fire points
        fire_pts = np.array([
            [230, 380], [250, 380 - flame_h], [270, 380 - int(flame_h * 0.7)], 
            [290, 380 - flame_h - 10], [310, 380 - int(flame_h * 0.6)], 
            [330, 380], [280, 400]
        ], np.int32)
        cv2.fillPoly(frame, [fire_pts], (0, 90, 255))
        
        core_pts = np.array([
            [250, 380], [270, 380 - int(flame_h * 0.8)], [290, 380 - int(flame_h * 0.6)], [300, 380]
        ], np.int32)
        cv2.fillPoly(frame, [core_pts], (0, 230, 255))

        # Camera OSD Alert
        cv2.putText(frame, "CAM-01 [SERVER ROOM 2] - **FLAME EVENT**", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "CAM-01 [SERVER ROOM 2] - LIVE STREAM", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)

    cv2.putText(frame, time_str, (420, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    out.write(frame)

out.release()
print(f"Sample benchmark CCTV video generated: {output_video_path} ({total_frames} frames, {duration_sec}s)")
