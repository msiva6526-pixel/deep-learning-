"""
SOC Metric Cards Component for FIREGUARD AI
"""

import streamlit as st
from utils.history_manager import history_manager

def render_metric_cards():
    """
    Renders 4 metric cards matching the SOC operations design:
      1. Total Images Analyzed
      2. Fire Detections (Red)
      3. Smoke Detections (Amber)
      4. Normal Images (Green)
    Metrics are tied to actual session detection records.
    """
    stats = history_manager.get_statistics()
    
    total = stats["total"]
    fire = stats["fire_count"]
    smoke = stats["smoke_count"]
    normal = stats["normal_count"]

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card" style="--card-accent: #38BDF8;">
                <div class="metric-label">
                    <span>Total Analyzed</span>
                    <span>📷</span>
                </div>
                <div class="metric-value">{total:,}</div>
                <div class="metric-sub">Surveillance telemetry logged</div>
            </div>
            <div class="metric-card" style="--card-accent: #EF4444;">
                <div class="metric-label">
                    <span style="color: #EF4444;">Fire Detections</span>
                    <span>🔥</span>
                </div>
                <div class="metric-value" style="color: #EF4444;">{fire:,}</div>
                <div class="metric-sub">Critical emergency alerts</div>
            </div>
            <div class="metric-card" style="--card-accent: #F59E0B;">
                <div class="metric-label">
                    <span style="color: #F59E0B;">Smoke Detections</span>
                    <span>⚠️</span>
                </div>
                <div class="metric-value" style="color: #F59E0B;">{smoke:,}</div>
                <div class="metric-sub">Early hazard warnings</div>
            </div>
            <div class="metric-card" style="--card-accent: #10B981;">
                <div class="metric-label">
                    <span style="color: #10B981;">Normal Images</span>
                    <span>✅</span>
                </div>
                <div class="metric-value" style="color: #10B981;">{normal:,}</div>
                <div class="metric-sub">Safe baseline frames</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
