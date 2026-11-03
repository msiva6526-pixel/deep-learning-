"""
FIREGUARD AI - Intelligent Fire & Smoke Detection in CCTV Surveillance
Deep Learning System using CNN with VGG16 Architecture
"""

import streamlit as st
from pathlib import Path
import config

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="FIREGUARD AI - Surveillance SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom SOC Dark Theme Stylesheet
def inject_custom_css():
    css_path = config.ASSETS_DIR / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

inject_custom_css()

# 3. Import Components & Pages
from components.sidebar import render_sidebar
from pages.dashboard import render_dashboard_page
from pages.detection import render_detection_page
from pages.monitoring import render_monitoring_page
from pages.history import render_history_page
from pages.analytics import render_analytics_page
from pages.model_info import render_model_info_page
from pages.settings import render_settings_page

# 4. Render Navigation Sidebar
current_page = render_sidebar()

# 5. Route to Selected SOC Page
if current_page == "Dashboard":
    render_dashboard_page()
elif current_page == "Fire & Smoke Detection":
    render_detection_page()
elif current_page == "CCTV Monitoring":
    render_monitoring_page()
elif current_page == "Detection History":
    render_history_page()
elif current_page == "Analytics & Reports":
    render_analytics_page()
elif current_page == "Model Information":
    render_model_info_page()
elif current_page == "Settings":
    render_settings_page()
else:
    render_dashboard_page()
