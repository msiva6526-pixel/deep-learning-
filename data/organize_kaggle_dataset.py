"""
Organizes downloaded Kaggle FIRE dataset into standard class directories:
  data/dataset/Fire/
  data/dataset/Normal/
"""

import shutil
from pathlib import Path
import os

target_dir = Path(__file__).resolve().parent / "dataset"
target_dir.mkdir(parents=True, exist_ok=True)

def organize_dataset(source_root: Path):
    print(f"Scanning source directory: {source_root}")
    fire_target = target_dir / "Fire"
    normal_target = target_dir / "Normal"
    fire_target.mkdir(exist_ok=True)
    normal_target.mkdir(exist_ok=True)

    fire_count = 0
    normal_count = 0

    for root, dirs, files in os.walk(source_root):
        rpath = Path(root)
        folder_name = rpath.name.lower()
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            src_file = rpath / f
            # Check if parent folder contains fire or non-fire
            if "non_fire" in folder_name or "non-fire" in folder_name or "normal" in folder_name or "neutral" in folder_name:
                dst = normal_target / f
                if not dst.exists():
                    shutil.copy2(src_file, dst)
                    normal_count += 1
            elif "fire" in folder_name or "flame" in folder_name:
                dst = fire_target / f
                if not dst.exists():
                    shutil.copy2(src_file, dst)
                    fire_count += 1

    print(f"Dataset Organized into '{target_dir}':")
    print(f"  • Fire images: {fire_count}")
    print(f"  • Normal images: {normal_count}")
    print(f"  • Total images: {fire_count + normal_count}")

if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache" / "kagglehub" / "datasets" / "phylake1337" / "fire-dataset"
    organize_dataset(src)
