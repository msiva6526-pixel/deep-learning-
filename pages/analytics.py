"""
Analytics & Intelligence Reports Page
"""

import streamlit as st
import pandas as pd
from utils.history_manager import history_manager
from components.charts import create_distribution_pie, create_activity_timeline, create_confidence_comparison_chart

def render_analytics_page():
    """Renders the comprehensive Deep Learning surveillance analytics."""
    st.markdown(
        """
        <div class="soc-header" style="margin-bottom: 1.5rem;">
            <div class="soc-title-group">
                <h1>ANALYTICS & INTELLIGENCE REPORTS</h1>
                <p>Quantitative Evaluation of Model Inferences & Temporal Telemetry</p>
            </div>
            <div>
                <span class="status-badge live"><span class="status-dot"></span>REPORT GENERATION ACTIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    records = history_manager.load_records()

    if not records:
        st.info("📊 No detection analytics available yet. Analyze CCTV frames to generate statistical charts and reports.")
        return

    # Statistical Overview Metrics
    stats = history_manager.get_statistics()
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Inferences", f"{stats['total']:,}")
    with col_m2:
        st.metric("Average Model Confidence", f"{stats['avg_confidence']:.1f}%")
    with col_m3:
        st.metric("Fire Detection Rate", f"{stats['emergency_rate']:.1f}%")
    with col_m4:
        st.metric("Normal Baseline Frames", f"{stats['normal_count']:,}")

    st.markdown("---")

    # Primary Analytics Charts
    col_ch1, col_ch2 = st.columns([1, 1.2])

    with col_ch1:
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="soc-panel-title"><span>🍩</span> Class Distribution</div>', unsafe_allow_html=True)
        donut_fig = create_distribution_pie(records)
        st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ch2:
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="soc-panel-title"><span>📊</span> Mean Confidence by Category</div>', unsafe_allow_html=True)
        conf_fig = create_confidence_comparison_chart(records)
        st.plotly_chart(conf_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Activity Timeline
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="soc-panel-title"><span>📈</span> Surveillance Event Activity Timeline</div>', unsafe_allow_html=True)
    timeline_fig = create_activity_timeline(records)
    st.plotly_chart(timeline_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Export Telemetry CSV
    col_down, _ = st.columns([1, 2])
    with col_down:
        csv_report = history_manager.export_to_csv()
        st.download_button(
            "📥 Download Surveillance Report (CSV)",
            data=csv_report,
            file_name="fireguard_analytics_report.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )

    # Statistical Context Disclaimer
    st.markdown(
        """
        <div class="safety-disclaimer">
            <strong>Statistical Clarification:</strong> The metrics and ratios above reflect the distribution of images uploaded 
            and evaluated within this operational session. They must not be conflated with the actual empirical rate or real-world incidence 
            of fires in physical infrastructure.
        </div>
        """,
        unsafe_allow_html=True
    )
