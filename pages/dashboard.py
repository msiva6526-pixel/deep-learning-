"""
Dashboard Page for FIREGUARD AI
"""

import streamlit as st
from datetime import datetime
import pandas as pd
import config
from utils.history_manager import history_manager
from models.model_loader import load_fire_model, get_model_summary_info
from components.metric_cards import render_metric_cards
from components.charts import create_distribution_pie, create_activity_timeline

def render_dashboard_page():
    """Renders the AI Fire Surveillance Operations Center Dashboard."""
    # Top Header
    now_str = datetime.now().strftime("%A, %d %B %Y | %H:%M:%S")
    model_status = load_fire_model()
    
    status_badge_html = (
        '<span class="status-badge live"><span class="status-dot"></span>ACTIVE MONITORING</span>'
        if model_status.is_loaded else
        '<span class="status-badge standby"><span class="status-dot"></span>STANDBY (NO WEIGHTS)</span>'
    )

    st.markdown(
        f"""
        <div class="soc-header">
            <div class="soc-title-group">
                <h1>AI FIRE SURVEILLANCE CENTER</h1>
                <p>Intelligent Fire and Smoke Detection Using Deep Learning (VGG16)</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 6px; font-family: monospace;">
                    {now_str}
                </div>
                {status_badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Metric Cards
    render_metric_cards()

    # Main Grid: Quick Action Banner & Analytics
    records = history_manager.load_records()

    col_charts_left, col_charts_right = st.columns([1, 1.2])

    with col_charts_left:
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="soc-panel-title"><span>📊</span> Detection Breakdown</div>', unsafe_allow_html=True)
        pie_fig = create_distribution_pie(records)
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_charts_right:
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="soc-panel-title"><span>📈</span> Surveillance Timeline</div>', unsafe_allow_html=True)
        timeline_fig = create_activity_timeline(records)
        st.plotly_chart(timeline_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Quick Jump Banner & Model Info Card
    col_act, col_model = st.columns([1.2, 1])
    
    with col_act:
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="soc-panel-title"><span>🚀</span> Quick Image Inspection</div>', unsafe_allow_html=True)
        st.write(
            "Upload any CCTV snapshot or select a pre-captured surveillance test frame "
            "to run automated neural inference via the VGG16 model."
        )
        if st.button("🔍 Launch Fire & Smoke Detection Engine", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "Fire & Smoke Detection"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_model:
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="soc-panel-title"><span>🧠</span> Active Model Telemetry</div>', unsafe_allow_html=True)
        meta = get_model_summary_info(model_status.model)
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #94A3B8;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Backbone:</span>
                    <strong style="color: #38BDF8;">VGG16 Transfer Learning</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Input Resolution:</span>
                    <strong style="color: #F8FAFC;">224 × 224 × 3</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span>Parameters:</span>
                    <strong style="color: #F8FAFC;">{meta['total_params']:,}</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Classes:</span>
                    <strong style="color: #10B981;">{', '.join(model_status.classes)}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Latest Detections Table
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>🕒</span> Recent Surveillance Activity</div>', unsafe_allow_html=True)
    
    if not records:
        st.info("No surveillance events recorded yet. Navigate to 'Fire & Smoke Detection' to analyze your first CCTV frame.")
    else:
        recent = records[:5]
        df_recent = pd.DataFrame([
            {
                "Detection ID": r["id"],
                "Timestamp": r["timestamp"],
                "Predicted Class": r["predicted_class"],
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Source": r["camera_id"],
                "Alert Status": r["alert_status"]
            }
            for r in recent
        ])
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
