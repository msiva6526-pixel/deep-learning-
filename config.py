"""
FIREGUARD AI - Central Configuration
Fire and Smoke Detection in CCTV Surveillance Using CNN with VGG16
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
SAMPLE_DIR = DATA_DIR / "sample_images"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "reports"
HISTORY_FILE = BASE_DIR / "detection_history.json"

# Create directories if they do not exist
for folder in [SAVED_MODELS_DIR, DATASET_DIR, SAMPLE_DIR, ASSETS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Default Model Weights Paths
DEFAULT_MODEL_PATH = SAVED_MODELS_DIR / "vgg16_fire_model.keras"
FALLBACK_H5_PATH = SAVED_MODELS_DIR / "vgg16_fire_model.h5"
CLASS_INDICES_PATH = SAVED_MODELS_DIR / "class_indices.json"
METRICS_PATH = SAVED_MODELS_DIR / "model_metrics.json"

# VGG16 Model Specifications
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
DEFAULT_CONFIDENCE_THRESHOLD = 0.65

# Default supported classes if dataset has 2 classes (Binary Fire/Normal)
# Or 3 classes (Fire, Smoke, Normal)
DEFAULT_CLASSES = ["Fire", "Normal"]
EXTENDED_CLASSES = ["Fire", "Smoke", "Normal"]

# Color Palette for UI & Charts
COLORS = {
    "bg_main": "#0B1020",
    "bg_sidebar": "#111827",
    "bg_card": "#151E32",
    "bg_card_hover": "#1E2A47",
    "primary": "#38BDF8",       # Electric Cyan
    "fire_alert": "#EF4444",     # Bright Red
    "smoke_warning": "#F59E0B",  # Amber
    "normal_success": "#10B981", # Emerald Green
    "border": "#22314E",
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8"
}

APP_TITLE = "FIREGUARD AI"
APP_SUBTITLE = "Intelligent Fire & Smoke Detection System Using CNN with VGG16"
APP_VERSION = "v2.4.0 (SOC Enterprise Edition)"
