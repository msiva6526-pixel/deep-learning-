"""
Detection History & Audit Trail Management Page
"""

import streamlit as st
import pandas as pd
from utils.history_manager import history_manager

def render_history_page():
    """Renders the searchable, filterable detection audit trail."""
    st.markdown(
        """
        <div class="soc-header" style="margin-bottom: 1.5rem;">
            <div class="soc-title-group">
                <h1>DETECTION AUDIT LOG & HISTORY</h1>
                <p>Verifiable Record of Deep Learning Inferences & Optical Telemetry</p>
            </div>
            <div>
                <span class="status-badge standby"><span class="status-dot"></span>IMMUTABLE AUDIT TRAIL</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    records = history_manager.load_records()

    if not records:
        st.info("ℹ️ No detection events have been recorded yet. Analyze CCTV frames to populate the audit log.")
        return

    # Filter and Search Controls
    col_search, col_filter, col_sort = st.columns([1.5, 1, 1])

    with col_search:
        search_query = st.text_input("🔍 Search Logs", placeholder="Search by ID, filename, or camera...")

    with col_filter:
        classes_present = sorted(list(set(r.get("predicted_class", "Normal") for r in records)))
        filter_class = st.selectbox("Filter by Class", ["All Classes"] + classes_present)

    with col_sort:
        sort_mode = st.selectbox("Sort Order", ["Newest First", "Oldest First", "Highest Confidence", "Lowest Confidence"])

    # Apply Filtering
    filtered = records
    if search_query:
        q = search_query.lower()
        filtered = [
            r for r in filtered
            if q in r.get("id", "").lower()
            or q in r.get("filename", "").lower()
            or q in r.get("camera_id", "").lower()
            or q in r.get("predicted_class", "").lower()
        ]

    if filter_class != "All Classes":
        filtered = [r for r in filtered if r.get("predicted_class") == filter_class]

    # Apply Sorting
    if sort_mode == "Newest First":
        filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    elif sort_mode == "Oldest First":
        filtered.sort(key=lambda x: x.get("timestamp", ""))
    elif sort_mode == "Highest Confidence":
        filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    elif sort_mode == "Lowest Confidence":
        filtered.sort(key=lambda x: x.get("confidence", 0))

    # Display Metrics Summary for filtered subset
    st.caption(f"Displaying **{len(filtered)}** of **{len(records)}** total audit records.")

    # Export & Clear History Buttons
    col_exp, col_clr = st.columns([1, 1])
    with col_exp:
        csv_data = history_manager.export_to_csv()
        st.download_button(
            "📥 Export Complete History as CSV",
            data=csv_data,
            file_name="fireguard_detection_audit_trail.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_clr:
        if "confirm_clear" not in st.session_state:
            st.session_state["confirm_clear"] = False

        if not st.session_state["confirm_clear"]:
            if st.button("🗑️ Clear Entire History", use_container_width=True):
                st.session_state["confirm_clear"] = True
                st.rerun()
        else:
            st.warning("Are you sure you want to permanently purge all detection logs?")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("⚠️ Confirm Purge", type="primary", use_container_width=True):
                    history_manager.clear_history()
                    st.session_state["confirm_clear"] = False
                    st.success("Detection history cleared.")
                    st.rerun()
            with c_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    # Table Visualization
    st.markdown("---")
    
    # Detailed Table Cards
    for rec in filtered:
        rec_id = rec.get("id")
        pclass = rec.get("predicted_class", "Normal")
        conf_pct = rec.get("confidence", 0) * 100
        tstamp = rec.get("timestamp")
        fname = rec.get("filename")
        cam = rec.get("camera_id")
        alert = rec.get("alert_status")

        accent_color = "#EF4444" if "fire" in pclass.lower() else ("#F59E0B" if "smoke" in pclass.lower() else "#10B981")
        
        with st.container():
            st.markdown(
                f"""
                <div style="background: #151E32; border: 1px solid #1E2A47; border-left: 4px solid {accent_color}; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-family: monospace; font-size: 0.85rem; color: #38BDF8; font-weight: 700;">{rec_id}</span>
                            <span style="margin: 0 8px; color: #475569;">|</span>
                            <strong style="color: {accent_color}; font-size: 0.95rem;">{pclass}</strong>
                            <span style="margin-left: 10px; font-family: monospace; font-size: 0.85rem; color: #F8FAFC;">{conf_pct:.2f}% Confidence</span>
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8; font-family: monospace;">
                            {tstamp}
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #94A3B8; margin-top: 6px;">
                        <span>Source: <strong style="color: #CBD5E1;">{cam}</strong> (File: {fname})</span>
                        <span style="text-transform: uppercase; font-weight: 700; color: {accent_color};">{alert}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            col_det1, col_det2 = st.columns([4, 1])
            with col_det2:
                if st.button("Delete Record", key=f"del_{rec_id}", use_container_width=True):
                    history_manager.delete_record(rec_id)
                    st.rerun()
