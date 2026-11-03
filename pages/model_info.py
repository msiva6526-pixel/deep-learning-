"""
Deep Learning Model Architecture & Dataset Specifications Page
"""

import streamlit as st
import json
from pathlib import Path
import config
from models.model_loader import load_fire_model, get_model_summary_info

def render_model_info_page():
    """Renders architectural specs, transfer learning workflow, and dataset documentation."""
    st.markdown(
        """
        <div class="soc-header" style="margin-bottom: 1.5rem;">
            <div class="soc-title-group">
                <h1>MODEL ARCHITECTURE & DATASET TELEMETRY</h1>
                <p>Transfer Learning Pipeline: VGG16 Convolutional Neural Network</p>
            </div>
            <div>
                <span class="status-badge live"><span class="status-dot"></span>DEEP CNN SPECIFICATION</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_status = load_fire_model()
    meta = get_model_summary_info(model_status.model)

    # Core Specifications Table
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>📋</span> Core Architectural Specifications</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            - **Model Name:** VGG16 (Visual Geometry Group 16-Layer)
            - **Technique:** Transfer Learning & Feature Extraction
            - **Framework:** TensorFlow / Keras 3.x
            - **Pretrained Weights Base:** ImageNet (1.4M benchmark images)
            - **Input Tensor Dimensions:** `(224, 224, 3)`
            """
        )
    with col2:
        st.markdown(
            f"""
            - **Dataset Source:** Kaggle FIRE Dataset
            - **Configured Output Classes:** `{', '.join(model_status.classes)}`
            - **Total Parameters:** `{meta['total_params']:,}`
            - **Weights Status:** `{'LOADED' if model_status.is_loaded else 'WEIGHTS MISSING'}`
            - **Weight Storage Path:** `{model_status.weights_path or 'None'}`
            """
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Workflow Diagram
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>📐</span> Neural Processing Workflow</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        ```
        [ CCTV Video Frame / Snapshot (RGB) ]
                         │
                         ▼
        [ Resizing & Interpolation: (224 × 224 × 3) ]
                         │
                         ▼
        [ VGG16 Preprocessing: RGB -> BGR & Zero-Centering Mean Subtraction ]
                         │
                         ▼
        ┌──────────────────────────────────────────────────────────┐
        │  VGG16 Convolutional Base (5 Blocks of 3×3 Convolutions) │
        │  (Pretrained ImageNet Weights - Frozen Feature Extractor) │
        └──────────────────────────────────────────────────────────┘
                         │
                         ▼
        [ Global Average Pooling 2D: Reduces 7×7×512 -> 512 Vector ]
                         │
                         ▼
        [ Dense Classification Layer: 256 Neurons (ReLU Activation) ]
                         │
                         ▼
        [ Batch Normalization + Dropout (Rate: 0.4 Regularization) ]
                         │
                         ▼
        [ Softmax Dense Head: Class Probabilities (Fire / Normal / Smoke) ]
        ```
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Training Metrics Section (Real vs Unavailable)
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>📈</span> Empirical Training & Validation Metrics</div>', unsafe_allow_html=True)

    if config.METRICS_PATH.exists():
        try:
            with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
                saved_metrics = json.load(f)
            
            st.success(f"Loaded verified metrics record from `{config.METRICS_PATH.name}`:")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Training Status", saved_metrics.get("status", "N/A"))
            with col_m2:
                st.metric("Validation Accuracy", f"{saved_metrics.get('val_accuracy', 0)}%")
            with col_m3:
                st.metric("Validation Loss", f"{saved_metrics.get('val_loss', 0):.4f}")
            with col_m4:
                st.metric("Epochs Completed", saved_metrics.get("epochs_completed", 0))

            st.caption(f"Last recorded training timestamp: {saved_metrics.get('trained_date')}")
        except Exception as e:
            st.error(f"Error reading metrics: {e}")
    else:
        st.info(
            "ℹ️ No training run has been executed yet on a local dataset partition. "
            "To measure empirical validation accuracy and loss, place your Kaggle dataset in 'data/dataset/' "
            "and execute the training pipeline below."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Training Instructions
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>🛠️</span> How to Fine-Tune on Kaggle FIRE Dataset</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        To train or fine-tune this model on your local Kaggle FIRE Dataset:

        1. **Download the Dataset:**
           - Download the **Kaggle FIRE Dataset** (e.g., from `kaggle.com/datasets/phylake1337/fire-dataset` or similar).
        2. **Organize Folders:**
           ```
           data/dataset/
           ├── Fire/
           │   ├── fire_001.jpg
           │   └── ...
           └── Normal/
               ├── normal_001.jpg
               └── ...
           ```
        3. **Run the Automated Training Script:**
           ```powershell
           & "C:\\Users\\Seilesh\\anaconda3\\python.exe" models/train_model.py --epochs 15 --batch_size 32
           ```
        4. The script will automatically:
           - Detect classes dynamically.
           - Partition training and validation subsets (80/20 split).
           - Train with early stopping and save optimal weights to `models/saved_models/vgg16_fire_model.keras`.
           - Record empirical accuracy and loss in `model_metrics.json`.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
