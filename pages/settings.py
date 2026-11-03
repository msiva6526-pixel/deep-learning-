"""
Application Settings & Configuration Page
"""

import streamlit as st
import config
from utils.history_manager import history_manager
from models.model_loader import load_fire_model

def render_settings_page():
    """Renders application settings, path diagnostics, and execution guide."""
    st.markdown(
        """
        <div class="soc-header" style="margin-bottom: 1.5rem;">
            <div class="soc-title-group">
                <h1>SYSTEM CONFIGURATION & SETTINGS</h1>
                <p>SOC Operational Parameters, Storage Paths & Environment Telemetry</p>
            </div>
            <div>
                <span class="status-badge standby"><span class="status-dot"></span>CONFIG SYNCED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_status = load_fire_model()

    # 1. Operational Parameters
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>⚙️</span> Operational Parameters</div>', unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("**Default Alert Confidence Threshold:**")
        st.info(f"Target: `{config.DEFAULT_CONFIDENCE_THRESHOLD * 100:.0f}%`")
        st.caption("Adjustable dynamically per inference on the Fire & Smoke Detection page.")

    with col_s2:
        st.write("**Active Trained Model Classes:**")
        st.code(f"{model_status.classes}")
        st.caption("⚠️ Note: True trained model classes are defined strictly by the model weights architecture.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. File System & Storage Diagnostics
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>📂</span> System Storage & Model Paths</div>', unsafe_allow_html=True)
    
    paths_data = [
        ("Base Application Root", str(config.BASE_DIR)),
        ("Model Weights Target", str(config.DEFAULT_MODEL_PATH)),
        ("Class Indices Mapping", str(config.CLASS_INDICES_PATH)),
        ("Empirical Metrics Log", str(config.METRICS_PATH)),
        ("Kaggle Dataset Directory", str(config.DATASET_DIR)),
        ("Sample Benchmark Frames", str(config.SAMPLE_DIR)),
        ("Audit History Storage", str(config.HISTORY_FILE))
    ]
    for label, pstr in paths_data:
        st.text(f"• {label}: {pstr}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Detection History Management
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>🗄️</span> Audit Trail Storage Operations</div>', unsafe_allow_html=True)
    
    rec_count = len(history_manager.load_records())
    st.write(f"Current audit record count: **{rec_count} entries**.")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        csv_data = history_manager.export_to_csv()
        st.download_button(
            "📥 Backup Audit Trail to CSV",
            data=csv_data,
            file_name="fireguard_audit_backup.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_a2:
        if st.button("🗑️ Reset All Detection Logs", use_container_width=True):
            history_manager.clear_history()
            st.success("History successfully cleared.")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Windows Execution Guide
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>💻</span> Windows Execution Instructions</div>', unsafe_allow_html=True)
    
    st.markdown(
        f"""
        To execute this project on Windows:

        ```powershell
        # 1. Open Terminal or Command Prompt in the project folder:
        cd "{config.BASE_DIR}"

        # 2. Launch the Streamlit SOC application:
        streamlit run app.py
        
        # Or using Anaconda Python directly:
        & "C:\\Users\\Seilesh\\anaconda3\\python.exe" -m streamlit run app.py
        ```

        **Application Version:** `{config.APP_VERSION}`
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
