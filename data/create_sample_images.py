"""
Generate realistic CCTV surveillance sample images for testing the detection pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sample_dir = Path(__file__).resolve().parent / "sample_images"
sample_dir.mkdir(parents=True, exist_ok=True)

# 1. CCTV Normal Server Room
img_normal = np.zeros((480, 640, 3), dtype=np.uint8)
# Background cool blue-gray
img_normal[:] = [30, 25, 20]  # BGR

# Server racks
for x in [60, 200, 340, 480]:
    cv2.rectangle(img_normal, (x, 100), (x + 100, 420), (45, 40, 35), -1)
    cv2.rectangle(img_normal, (x, 100), (x + 100, 420), (70, 65, 55), 2)
    # Server LED lights
    for y in range(120, 400, 25):
        cv2.circle(img_normal, (x + 20, y), 3, (255, 180, 0), -1) # Blueish LED
        cv2.circle(img_normal, (x + 40, y), 3, (0, 255, 100), -1) # Green LED

# Ceiling lights
cv2.rectangle(img_normal, (150, 20), (490, 40), (220, 230, 240), -1)

# CCTV OSD overlay
cv2.putText(img_normal, "CAM-03 [SERVER ROOM 2] - LIVE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 1, cv2.LINE_AA)
cv2.putText(img_normal, "2026-10-29 00:15:30", (460, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

cv2.imwrite(str(sample_dir / "cctv_normal_room.jpg"), img_normal)

# 2. CCTV Fire Incident Scene
img_fire = img_normal.copy()
# Draw fire flames in the middle server rack
fire_pts = np.array([
    [230, 380], [250, 290], [270, 330], [290, 260],
    [310, 320], [330, 380], [280, 400]
], np.int32)
# Orange outer flame
cv2.fillPoly(img_fire, [fire_pts], (0, 69, 255))
# Inner yellow core
core_pts = np.array([
    [250, 380], [270, 310], [290, 330], [300, 380]
], np.int32)
cv2.fillPoly(img_fire, [core_pts], (0, 215, 255))
# Fire glow
cv2.circle(img_fire, (280, 350), 110, (0, 60, 180), -1)
# Re-apply flames on top of glow
cv2.fillPoly(img_fire, [fire_pts], (0, 100, 255))
cv2.fillPoly(img_fire, [core_pts], (0, 240, 255))

cv2.putText(img_fire, "CAM-01 [WAREHOUSE ZONE B] - ALERT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
cv2.putText(img_fire, "2026-10-29 00:16:04", (460, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

cv2.imwrite(str(sample_dir / "cctv_fire_scene.jpg"), img_fire)

# 3. CCTV Smoke Incident Scene
img_smoke = img_normal.copy()
# Draw semi-transparent gray smoke haze
smoke_overlay = img_smoke.copy()
cv2.circle(smoke_overlay, (320, 240), 180, (190, 190, 190), -1)
cv2.circle(smoke_overlay, (240, 200), 140, (170, 170, 170), -1)
cv2.circle(smoke_overlay, (400, 220), 130, (180, 180, 180), -1)
img_smoke = cv2.addWeighted(smoke_overlay, 0.55, img_smoke, 0.45, 0)

cv2.putText(img_smoke, "CAM-02 [CORRIDOR NORTH] - WARNING", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)
cv2.putText(img_smoke, "2026-10-29 00:16:45", (460, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

cv2.imwrite(str(sample_dir / "cctv_smoke_scene.jpg"), img_smoke)

print(f"Generated 3 sample benchmark CCTV images in {sample_dir}")
