"""Generate FIREGUARD AI flame & shield motif logo in high-res PNG format"""
from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path

assets_dir = Path(__file__).resolve().parent
output_path = assets_dir / "logo.png"

# Canvas dimensions
size = (512, 512)
img = Image.new("RGBA", size, (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Outer glow / Shield outline
shield_points = [
    (256, 40),   # top center
    (430, 90),   # top right
    (390, 330),  # mid right curve
    (256, 470),  # bottom tip
    (122, 330),  # mid left curve
    (82, 90)     # top left
]

# Draw shield background with SOC navy gradient fill
draw.polygon(shield_points, fill=(17, 24, 39, 245), outline=(56, 189, 248, 255), width=6)

# Inner shield border
inner_shield = [
    (256, 65),
    (405, 110),
    (370, 315),
    (256, 440),
    (142, 315),
    (107, 110)
]
draw.polygon(inner_shield, fill=(21, 30, 50, 230), outline=(30, 42, 71, 200), width=3)

# Draw stylized flame inside
# Outer flame (Bright Fire Red/Orange)
flame_outer = [
    (256, 130), # flame tip
    (300, 200),
    (330, 260),
    (310, 340),
    (280, 380),
    (256, 395),
    (232, 380),
    (202, 340),
    (182, 260),
    (212, 200)
]
draw.polygon(flame_outer, fill=(239, 68, 68, 240))

# Mid flame (Amber / Gold)
flame_mid = [
    (256, 175),
    (290, 235),
    (305, 290),
    (285, 360),
    (256, 375),
    (227, 360),
    (207, 290),
    (222, 235)
]
draw.polygon(flame_mid, fill=(245, 158, 11, 250))

# Core flame (Electric Cyan / White Hot Core)
flame_core = [
    (256, 230),
    (275, 280),
    (280, 325),
    (256, 350),
    (232, 325),
    (237, 280)
]
draw.polygon(flame_core, fill=(56, 189, 248, 255))

# Center tiny white spark
draw.ellipse([(251, 310), (261, 330)], fill=(255, 255, 255, 255))

# Save image
img.save(output_path, "PNG")
print(f"Logo generated successfully at: {output_path}")
