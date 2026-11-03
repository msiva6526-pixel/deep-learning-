"""
Custom SOC Sidebar Navigation Component for FIREGUARD AI
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import config
from models.model_loader import load_fire_model

def render_sidebar() -> str:
    """
    Renders the custom SOC sidebar with logo, navigation, system telemetry, and footer.
    Returns:
        str: Currently selected page name
    """
    with st.sidebar:
        # Logo & Brand Header
        logo_path = config.ASSETS_DIR / "logo.png"
        col_logo, col_title = st.columns([1, 2.5])
        
        with col_logo:
            if logo_path.exists():
                st.image(str(logo_path), width=64)
            else:
                st.markdown("🛡️🔥", unsafe_allow_html=True)
                
        with col_title:
            st.markdown(
                """
                <div style="line-height: 1.2; padding-top: 5px;">
                    <span style="font-size: 1.25rem; font-weight: 800; color: #38BDF8; letter-spacing: 0.05em;">FIREGUARD AI</span><br>
                    <span style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">Surveillance SOC</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # System Status Indicator
        model_status = load_fire_model()
        status_color = "#10B981" if model_status.is_loaded else "#F59E0B"
        status_text = "SURVEILLANCE ACTIVE" if model_status.is_loaded else "STANDBY / NO WEIGHTS"
        
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid #1E2A47; border-radius: 8px; padding: 10px 14px; margin: 18px 0 20px 0;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background-color: {status_color}; box-shadow: 0 0 8px {status_color};"></span>
                        <span style="font-size: 0.75rem; font-weight: 700; color: {status_color}; letter-spacing: 0.04em;">{status_text}</span>
                    </div>
                    <span style="font-size: 0.7rem; color: #64748B; font-family: monospace;">VGG16</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                OPERATIONS MENU
            </div>
            """,
            unsafe_allow_html=True
        )

        # Navigation Options
        pages = [
            ("Dashboard", "📊"),
            ("Fire & Smoke Detection", "🔥"),
            ("CCTV Monitoring", "📹"),
            ("Detection History", "🗂️"),
            ("Analytics & Reports", "📈"),
            ("Model Information", "🧠"),
            ("Settings", "⚙️")
        ]

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Dashboard"

        # Navigation Buttons with highlight
        for page_name, icon in pages:
            is_active = st.session_state["current_page"] == page_name
            btn_label = f"{icon}  {page_name}"
            
            if st.button(
                btn_label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state["current_page"] = page_name
                st.rerun()

        st.markdown("---")

        # Telemetry Quick Card
        st.markdown(
            f"""
            <div style="background: #0B1020; border: 1px solid #1E2A47; border-radius: 8px; padding: 12px; font-size: 0.75rem; color: #94A3B8;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Input Resolution:</span>
                    <strong style="color: #F8FAFC;">224 × 224</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Classes:</span>
                    <strong style="color: #38BDF8;">{', '.join(model_status.classes)}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Framework:</span>
                    <strong style="color: #F8FAFC;">Keras / TF 2.21</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Compact Footer
        st.markdown(
            f"""
            <div style="padding-top: 20px; text-align: center; font-size: 0.7rem; color: #475569;">
                <div>{config.APP_TITLE} {config.APP_VERSION}</div>
                <div style="margin-top: 4px;">Security Operations Center Interface</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    return st.session_state["current_page"]
