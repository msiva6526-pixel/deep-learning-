"""
CCTV Multi-Channel Surveillance Monitoring Page
"""

import streamlit as st
from PIL import Image
from datetime import datetime
from pathlib import Path
import config
from models.model_loader import load_fire_model
from models.predict import run_vgg16_inference
from utils.history_manager import history_manager

def render_monitoring_page():
    """Renders the Multi-Camera Surveillance Matrix."""
    st.markdown(
        """
        <div class="soc-header" style="margin-bottom: 1.5rem;">
            <div class="soc-title-group">
                <h1>CCTV SURVEILLANCE MATRIX</h1>
                <p>Multi-Channel Camera Monitoring & Frame Analysis Operations</p>
            </div>
            <div>
                <span class="status-badge live"><span class="status-dot"></span>4 CHANNELS SYNCHRONIZED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_status = load_fire_model()

    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1E2A47; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; font-size: 0.85rem; color: #94A3B8;">
            ℹ️ <strong>Surveillance Operations Note:</strong> This monitoring room provides channel management, camera snapshot diagnostics, 
            and real-time optical frame testing. Feed snapshots can be analyzed individually with the deep learning VGG16 model.
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_matrix, tab_webcam = st.tabs(["🖥️ Quad-Camera Matrix Layout", "📸 Live Webcam Frame Capture"])

    with tab_matrix:
        # Camera Definitions
        cameras = [
            {
                "id": "CAM-01",
                "name": "Warehouse Zone B",
                "sample": config.SAMPLE_DIR / "cctv_fire_scene.jpg",
                "status": "ALERT",
                "fps": 30.0,
                "badge_color": "#EF4444"
            },
            {
                "id": "CAM-02",
                "name": "North Corridor",
                "sample": config.SAMPLE_DIR / "cctv_smoke_scene.jpg",
                "status": "WARNING",
                "fps": 29.8,
                "badge_color": "#F59E0B"
            },
            {
                "id": "CAM-03",
                "name": "Server Facility Room 2",
                "sample": config.SAMPLE_DIR / "cctv_normal_room.jpg",
                "status": "NORMAL",
                "fps": 30.0,
                "badge_color": "#10B981"
            },
            {
                "id": "CAM-04",
                "name": "Main Perimeter Entry",
                "sample": config.SAMPLE_DIR / "cctv_normal_room.jpg",
                "status": "NORMAL",
                "fps": 25.0,
                "badge_color": "#10B981"
            }
        ]

        # 2x2 Camera Grid
        col_c1, col_c2 = st.columns(2)
        col_c3, col_c4 = st.columns(2)
        col_slots = [col_c1, col_c2, col_c3, col_c4]

        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for idx, cam in enumerate(cameras):
            with col_slots[idx]:
                st.markdown(
                    f"""
                    <div class="cctv-feed-card">
                        <div class="cctv-feed-header">
                            <div>
                                <span class="cctv-rec-dot"></span>
                                <strong>{cam['id']}</strong> [{cam['name']}]
                            </div>
                            <span style="color: {cam['badge_color']}; font-weight: 700;">{cam['status']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Check if camera has custom snapshot uploaded or fallback to sample
                sample_p = cam["sample"]
                if sample_p.exists():
                    st.image(str(sample_p), use_container_width=True)
                else:
                    st.info(f"Awaiting feed stream for {cam['id']}")

                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; font-family: monospace; margin: 4px 0 8px 0;">
                        <span>FPS: {cam['fps']}</span>
                        <span>{now_time}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Individual camera analysis button
                if st.button(f"🔍 Analyze {cam['id']} Frame", key=f"btn_analyze_{cam['id']}", use_container_width=True):
                    if not model_status.is_loaded:
                        st.error("Cannot perform analysis: VGG16 model weights not loaded.")
                    elif sample_p.exists():
                        with st.spinner(f"Analyzing {cam['id']} snapshot..."):
                            res = run_vgg16_inference(str(sample_p), model_status)
                            if res["success"]:
                                pclass = res["predicted_class"]
                                conf_pct = res["confidence_percentage"]
                                
                                # Save to history
                                history_manager.add_record(
                                    predicted_class=pclass,
                                    confidence=res["confidence"],
                                    filename=sample_p.name,
                                    camera_id=f"{cam['id']} - {cam['name']}",
                                    notes="CCTV Matrix Snapshot",
                                    details=res
                                )

                                if "fire" in pclass.lower():
                                    st.error(f"🚨 {cam['id']}: FIRE DETECTED ({conf_pct}% confidence)")
                                elif "smoke" in pclass.lower():
                                    st.warning(f"⚠️ {cam['id']}: SMOKE DETECTED ({conf_pct}% confidence)")
                                else:
                                    st.success(f"✅ {cam['id']}: ALL CLEAR ({conf_pct}% confidence)")
                            else:
                                st.error(res["message"])

    with tab_webcam:
        st.markdown("#### Connect Local Optical Sensor / Webcam Frame")
        st.write("You can test real frame capture directly from your computer's webcam or optical sensor:")
        
        camera_img = st.camera_input("Capture frame from local optical sensor", key="webcam_capture")
        if camera_img is not None:
            pil_cam = Image.open(camera_img).convert("RGB")
            st.image(pil_cam, caption="Captured Optical Frame", width=360)
            
            if st.button("🔥 Run VGG16 Inference on Optical Frame", type="primary"):
                if not model_status.is_loaded:
                    st.error("VGG16 model weights are not loaded.")
                else:
                    with st.spinner("Processing optical frame through VGG16..."):
                        webcam_res = run_vgg16_inference(pil_cam, model_status)
                        if webcam_res["success"]:
                            pred = webcam_res["predicted_class"]
                            conf_pct = webcam_res["confidence_percentage"]
                            
                            history_manager.add_record(
                                predicted_class=pred,
                                confidence=webcam_res["confidence"],
                                filename="optical_sensor_capture.jpg",
                                camera_id="Local Optical Sensor",
                                notes=f"Latency: {webcam_res['latency_ms']}ms",
                                details=webcam_res
                            )

                            if "fire" in pred.lower():
                                st.error(f"🚨 CRITICAL: FIRE DETECTED ({conf_pct}%)")
                            elif "smoke" in pred.lower():
                                st.warning(f"⚠️ WARNING: SMOKE DETECTED ({conf_pct}%)")
                            else:
                                st.success(f"✅ CLEAR: NO FIRE DETECTED ({conf_pct}%)")
                                
                            st.json(webcam_res["all_probabilities"])
                        else:
                            st.error(webcam_res["message"])
