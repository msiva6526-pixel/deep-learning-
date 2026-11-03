# FIREGUARD AI: Intelligent Fire & Smoke Detection in CCTV Surveillance Using CNN with VGG16

[![Deep Learning](https://img.shields.io/badge/Architecture-VGG16%20Transfer%20Learning-38BDF8?style=for-the-badge)](https://keras.io)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras%203-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit%20SOC%20Design-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Analytics](https://img.shields.io/badge/Analytics-Plotly%20Interactive-10B981?style=for-the-badge)](https://plotly.com)

---

## 1. Project Introduction

**FIREGUARD AI** is an enterprise-grade artificial intelligence surveillance and emergency monitoring platform designed to autonomously identify **fire** and **smoke** hazards in closed-circuit television (CCTV) feeds. 

By replacing rudimentary color-heuristic techniques with a deep **Convolutional Neural Network (CNN) powered by VGG16 transfer learning**, FIREGUARD AI extracts multi-scale visual features—ranging from early smoke vapor gradients to acute flame edge signatures—delivering reliable detection with millisecond latency.

The system is paired with a custom-engineered **Security Operations Center (SOC) dark-themed interface** (`#0B1020`), offering operational dashboards, quad-channel CCTV matrix monitoring, live webcam inference, verifiable audit history, and quantitative analytics.

---

## 2. Main Features

- 🛡️ **Custom SOC Dark Dashboard**: Real-time operational telemetry, live system status indicators, incident distribution donut charts, and temporal activity timelines.
- 🔥 **Deep Learning Detection Engine**: Genuine VGG16 transfer learning inference with confidence scoring, threshold tuning, and class probability distributions.
- 🎥 **CCTV Video Stream Analysis**: Upload MP4, AVI, MOV recordings or run benchmark surveillance videos with automated frame-by-frame inference, temporal hazard timeline charts, and incident keyframe extraction.
- 📹 **CCTV Multi-Channel Monitoring**: Simulated 4-camera matrix layout (Warehouse, North Corridor, Server Facility, Perimeter Entry) with individual frame analysis and local optical sensor / webcam integration.
- 🗂️ **Verifiable Detection History**: Searchable, filterable audit trail tracking Detection ID, timestamps, confidence scores, source cameras, and alert levels (no fabricated data).
- 📈 **Interactive Plotly Analytics**: Real-time class breakdown, mean confidence comparisons, timeline trend evaluation, and one-click CSV report export.
- 🧠 **Model Information & Diagnostics**: Complete VGG16 layer breakdown, workflow diagram, parameter counts, and measured validation metrics tracking.
- 🎯 **Preset Benchmark Test Frames**: Built-in CCTV test scenes (Fire Incident, Smoke Dispersion, Normal Server Room) for immediate evaluation without requiring external image downloads.

---

## 3. Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10 - 3.13 | Core programming language |
| **Deep Learning** | VGG16 (ImageNet Backbone) | CNN feature extraction with custom classification head |
| **Framework** | TensorFlow 2.x & Keras 3.x | Tensor computation, transfer learning, and model persistence |
| **Computer Vision** | OpenCV & Pillow (PIL) | CCTV frame decoding, resizing (224×224), and zero-centering |
| **Web Frontend** | Streamlit + Custom CSS | Modern SOC dark theme interface (`#0B1020`, `#38BDF8`, `#EF4444`) |
| **Data Processing** | NumPy & Pandas | Array manipulation and audit log management |
| **Interactive Charts** | Plotly Express & Graph Objects | Dynamic class distribution and incident timeline charts |
| **Dataset Support** | Kaggle FIRE Dataset | Binary (Fire / Normal) and Multi-Class (Fire / Smoke / Normal) |

---

## 4. Dataset Download & Organization

The system is configured to support the **Kaggle FIRE Dataset** (such as [Kaggle Fire Dataset by Phylake1337](https://www.kaggle.com/datasets/phylake1337/fire-dataset) or similar benchmark collections).

### Directory Structure Requirements:
Place your images into `data/dataset/` with subfolders named according to the target classes:

```
data/
└── dataset/
    ├── Fire/
    │   ├── fire_frame_001.jpg
    │   ├── fire_frame_002.jpg
    │   └── ...
    └── Normal/
        ├── normal_frame_001.jpg
        ├── normal_frame_002.jpg
        └── ...
```

*Note: If your dataset includes an additional class such as `Smoke`, simply create a `data/dataset/Smoke/` subfolder. The training script automatically discovers and adapts the model output head to match your classes.*

---

## 5. Model Training & Initialization Steps

### Option A: Initialize Baseline VGG16 Model (Immediate Testing)
To generate the pre-compiled VGG16 transfer model with ImageNet feature extraction weights:
```powershell
python models/train_model.py --init_baseline
```

### Option B: Fine-Tune on Your Kaggle FIRE Dataset
To run complete fine-tuning across your training data:
```powershell
python models/train_model.py --data_dir data/dataset --epochs 15 --batch_size 32
```
This script automatically:
1. Validates and discovers class subdirectories.
2. Applies 80/20 train-validation split with data streaming.
3. Attaches `GlobalAveragePooling2D`, `Dense(256)`, `BatchNormalization`, `Dropout(0.4)`, and `Dense(num_classes, softmax)`.
4. Executes Adam optimization with `EarlyStopping` and `ModelCheckpoint`.
5. Saves verified weights to `models/saved_models/vgg16_fire_model.keras` and logs real measured metrics to `model_metrics.json`.

---

## 6. Installation Guide (Windows)

### Step 1: Open Terminal / Command Prompt
Open PowerShell or Command Prompt and navigate to the project directory:
```powershell
cd "c:\Users\Seilesh\Downloads\DEEP LEARNING"
```

### Step 2: (Optional) Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```
*(If using an existing Anaconda environment, run using `& "C:\Users\Seilesh\anaconda3\python.exe" -m pip install -r requirements.txt`)*

---

## 7. Application Execution Commands

Run the Streamlit SOC application:

```powershell
streamlit run app.py
```

Or via Anaconda Python directly:
```powershell
& "C:\Users\Seilesh\anaconda3\python.exe" -m streamlit run app.py
```

The application will launch automatically in your default browser at:
**`http://localhost:8501`**

---

## 8. Project Folder Structure

```
DEEP LEARNING/
├── app.py                      # Main application entry point & page router
├── config.py                   # Central paths, thresholds, and theme constants
├── requirements.txt            # Package dependencies
├── README.md                   # Complete documentation
│
├── assets/
│   ├── styles.css              # Custom SOC dark-theme CSS design system
│   ├── logo.png                # Flame and shield motif vector logo
│   └── generate_logo.py        # Logo generation utility
│
├── models/
│   ├── model_loader.py         # Model loader with @st.cache_resource
│   ├── predict.py              # Inference pipeline & probability computation
│   ├── train_model.py          # VGG16 transfer learning training script
│   └── saved_models/
│       ├── vgg16_fire_model.keras  # Fine-tuned VGG16 model weights
│       ├── class_indices.json      # Dynamic class mapping
│       └── model_metrics.json      # Verified training and validation metrics
│
├── components/
│   ├── sidebar.py              # SOC sidebar navigation & telemetry
│   ├── metric_cards.py         # 4 KPI operational metric cards
│   ├── charts.py               # Interactive Plotly visualization charts
│   └── upload_component.py     # Image uploader & benchmark frame selector
│
├── pages/
│   ├── dashboard.py            # Overview dashboard & recent detections
│   ├── detection.py            # Primary image inference interface
│   ├── monitoring.py           # CCTV 4-channel matrix & optical frame capture
│   ├── history.py              # Filterable audit logs & CSV export
│   ├── analytics.py            # Quantitative intelligence reports
│   ├── model_info.py           # Architecture specs & layer workflow
│   └── settings.py             # System paths & operational thresholds
│
├── utils/
│   ├── preprocessing.py        # VGG16 BGR mean subtraction & resizing (224x224)
│   ├── validators.py           # Image format, size, and integrity checks
│   └── history_manager.py      # Persistent JSON audit trail & statistics engine
│
└── data/
    ├── sample_images/          # Benchmark test CCTV frames (Fire, Smoke, Normal)
    └── dataset/                # Kaggle FIRE dataset storage directory
```

---

## 9. Common Errors and Solutions

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Model weights not found` | No `.keras` file in `models/saved_models/` | Run `python models/train_model.py --init_baseline` to generate baseline VGG16 weights. |
| `Unsupported file type (.xyz)` | User uploaded an invalid file format | Ensure images are in JPG, JPEG, or PNG format. |
| `Corrupted image file` | Incomplete or corrupted image download | Re-download or open the file in Paint/Photos to re-save. |
| `Port 8501 is already in use` | Another Streamlit instance is running | Run `streamlit run app.py --server.port 8502`. |
| `CUDA GPU not detected` | Native Windows TensorFlow >= 2.11 CPU mode | VGG16 inference executes efficiently on CPU; GPU on Windows requires WSL2. |

---

## 10. Engineering Ethics & Safety Disclaimer

Model predictions produced by FIREGUARD AI are statistical inferences derived from optical surveillance patterns. While engineered to achieve high precision and low false-positive rates, automated AI classifications must serve as decision support tools and must be integrated alongside certified thermal/smoke sensors and trained human monitoring personnel.
#   d e e p - l e a r n i n g -  
 