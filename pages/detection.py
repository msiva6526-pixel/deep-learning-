"""
Primary Fire and Smoke Detection Page with Direct Manual Video & Image Upload
"""

import streamlit as st
import time
import json
import tempfile
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import plotly.graph_objects as go

import config
from components.upload_component import render_upload_selector
from models.model_loader import load_fire_model
from models.predict import run_vgg16_inference
from utils.video_processor import analyze_cctv_video
from utils.history_manager import history_manager

def render_detection_page():
    """Renders the main Deep Learning inference interface for manual video and image analysis."""
    st.markdown(
        """
        <div class="soc-header" style="margin-bottom: 1.5rem;">
            <div class="soc-title-group">
                <h1>FIRE & SMOKE DETECTION ENGINE</h1>
                <p>Manual CCTV Video & Image Stream Surveillance using CNN with VGG16</p>
            </div>
            <div>
                <span class="status-badge live"><span class="status-dot"></span>INFERENCE READY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_status = load_fire_model()
    
    # Model Availability Alert
    if not model_status.is_loaded:
        st.warning(
            f"⚠️ **Model Notice**: {model_status.status_message}\n\n"
            "You can initialize baseline weights by running: `python models/train_model.py --init_baseline`"
        )

    # Main Mode Tabs - Video Analysis First
    tab_video, tab_image = st.tabs(["🎥 Upload & Analyze CCTV Video", "📷 Upload CCTV Image"])

    # =========================================================================
    # TAB 1: CCTV VIDEO ANALYSIS (DIRECT MANUAL UPLOAD)
    # =========================================================================
    with tab_video:
        _render_video_detection(model_status)

    # =========================================================================
    # TAB 2: SINGLE IMAGE ANALYSIS
    # =========================================================================
    with tab_image:
        _render_image_detection(model_status)


def _render_video_detection(model_status):
    """Renders direct manual video upload and frame-by-frame temporal analysis."""
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">
                UPLOAD CCTV VIDEO FILE
            </h3>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #94A3B8;">
                Upload your recorded CCTV video stream (.mp4, .avi, .mov, .mkv) to perform automated neural fire & smoke classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Select and upload CCTV video recording",
        type=["mp4", "avi", "mov", "mkv"],
        key="manual_video_uploader",
        help="Upload standard CCTV surveillance video formats: MP4, AVI, MOV, MKV"
    )

    video_source_path = None
    video_filename = None

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_video.name}")
        tfile.write(uploaded_video.read())
        video_source_path = tfile.name
        video_filename = uploaded_video.name
        tfile.close()

    st.markdown('</div>', unsafe_allow_html=True)

    if not video_source_path:
        st.info("👆 Please upload a CCTV video file using the upload area above to begin analysis.")
        return

    # Video Player Preview & Settings
    col_v1, col_v2 = st.columns([1.2, 1])
    with col_v1:
        st.video(video_source_path)
    with col_v2:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E2A47; border-radius: 8px; padding: 14px; font-size: 0.85rem; margin-bottom: 12px;">
                <div style="color: #38BDF8; font-weight: 700; margin-bottom: 8px;">ACTIVE VIDEO FILE</div>
                <div style="color: #F8FAFC; margin-bottom: 4px;"><strong>Filename:</strong> {video_filename}</div>
                <div style="color: #10B981;">✓ File loaded and ready for neural temporal inspection</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        sample_rate = st.select_slider(
            "Temporal Frame Sampling (FPS)",
            options=[1.0, 2.0, 4.0, 5.0],
            value=2.0,
            help="Higher FPS samples more frames per second of video for finer precision."
        )
        video_conf_thresh = st.slider(
            "Alert Sensitivity Threshold",
            min_value=0.40,
            max_value=0.95,
            value=config.DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            key="vid_conf_thresh_manual"
        )
        btn_start_video = st.button("🎬 ANALYZE CCTV VIDEO STREAM", type="primary", use_container_width=True, key="btn_run_video_proc")

    # Execution
    if btn_start_video:
        if not model_status.is_loaded:
            st.error("Cannot perform inference: VGG16 model weights not loaded.")
            return

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_prog(p, msg):
            progress_bar.progress(p)
            status_text.text(msg)

        vid_result = analyze_cctv_video(
            video_source_path,
            model_status,
            target_fps_sample=sample_rate,
            conf_threshold=video_conf_thresh,
            progress_callback=on_prog
        )

        progress_bar.empty()
        status_text.empty()

        if not vid_result["success"]:
            st.error(vid_result["message"])
            return

        st.session_state["latest_video_result"] = vid_result

    # Display Video Analysis Results
    if "latest_video_result" in st.session_state:
        vr = st.session_state["latest_video_result"]
        has_fire = vr["has_fire"]

        # Alert Banner
        if has_fire:
            st.markdown(
                f"""
                <div class="alert-card-fire">
                    <div class="alert-header">
                        <span style="font-size: 2.8rem;">🚨</span>
                        <div>
                            <h2 class="alert-title">CRITICAL HAZARD: FIRE DETECTED IN VIDEO</h2>
                            <div style="font-size: 0.85rem; color: #94A3B8;">
                                First Detected At: <strong style="color: #EF4444;">T+{vr['first_fire_sec']}s</strong> | 
                                Peak Fire Confidence: <strong style="color: #EF4444;">{vr['peak_fire_confidence']}%</strong> | 
                                Fire Frames: <strong style="color: #F8FAFC;">{vr['fire_frames']} of {vr['analyzed_frames']}</strong>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="alert-card-normal">
                    <div class="alert-header">
                        <span style="font-size: 2.8rem;">✅</span>
                        <div>
                            <h2 class="alert-title">VIDEO STREAM VERIFIED: NO FIRE DETECTED</h2>
                            <div style="font-size: 0.85rem; color: #94A3B8;">
                                All sampled frames tested below emergency thresholds. Surveillance facility baseline safe.
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 4 Metric Cards for Video
        col_vm1, col_vm2, col_vm3, col_vm4 = st.columns(4)
        with col_vm1:
            st.metric("Video Duration", f"{vr['duration_sec']}s")
        with col_vm2:
            st.metric("Frames Analyzed", f"{vr['analyzed_frames']} frames")
        with col_vm3:
            st.metric("Processing Latency", f"{vr['total_processing_time_sec']}s")
        with col_vm4:
            st.metric("Neural Throughput", f"{vr['processing_fps']} FPS")

        # Temporal Timeline Chart (Plotly)
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="soc-panel-title"><span>📈</span> Temporal Hazard Timeline</div>', unsafe_allow_html=True)

        tl = vr.get("timeline", [])
        if tl:
            seconds_x = [t["second"] for t in tl]
            fire_probs = [t["fire_probability"] for t in tl]

            fig_v = go.Figure()
            # Threshold horizontal line
            fig_v.add_hline(
                y=video_conf_thresh * 100,
                line_dash="dot",
                line_color="#EF4444",
                annotation_text=f"Alert Threshold ({int(video_conf_thresh*100)}%)",
                annotation_position="bottom right"
            )
            # Area fill for fire confidence
            fig_v.add_trace(go.Scatter(
                x=seconds_x,
                y=fire_probs,
                mode="lines+markers",
                name="Fire Confidence (%)",
                line=dict(color="#EF4444", width=3),
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.2)",
                marker=dict(size=6, color="#EF4444")
            ))
            fig_v.update_layout(
                paper_bgcolor="#151E32",
                plot_bgcolor="#151E32",
                font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
                xaxis=dict(title="Video Timeline (Seconds)", gridcolor="#1E2A47", zeroline=False),
                yaxis=dict(title="Fire Confidence (%)", gridcolor="#1E2A47", range=[0, 105], zeroline=False),
                height=320,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(fig_v, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Keyframe Gallery
        kfs = vr.get("keyframes", [])
        if kfs:
            st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
            st.markdown('<div class="soc-panel-title"><span>🖼️</span> Keyframe Incident Gallery</div>', unsafe_allow_html=True)
            cols_kf = st.columns(min(4, len(kfs)))
            for idx, kf in enumerate(kfs[:8]):
                with cols_kf[idx % 4]:
                    st.image(kf["image"], caption=f"T+{kf['second']}s | {kf['class']} ({kf['confidence']:.1f}%)", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Action: Save Video Incident to History
        if st.button("💾 Save Video Analysis Summary to Detection History", type="primary", key="save_vid_history_btn"):
            rec_class = "Fire" if has_fire else "Normal"
            saved = history_manager.add_record(
                predicted_class=rec_class,
                confidence=vr["peak_fire_confidence"] / 100.0 if has_fire else 0.85,
                filename=vr["filename"],
                camera_id="CCTV Video Stream",
                notes=f"Duration: {vr['duration_sec']}s, Peak Conf: {vr['peak_fire_confidence']}%, First Fire: T+{vr['first_fire_sec']}s",
                details=vr
            )
            st.success(f"Video analysis audit record [{saved['id']}] saved to detection history!")


def _render_image_detection(model_status):
    """Renders manual single image frame detection."""
    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
    img_pil, filename = render_upload_selector()
    st.markdown('</div>', unsafe_allow_html=True)

    if img_pil is None:
        st.info("👆 Please upload a CCTV snapshot above to begin analysis.")
        return

    # Confidence Threshold Slider
    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        conf_thresh = st.slider(
            "Alert Sensitivity Threshold",
            min_value=0.40,
            max_value=0.95,
            value=config.DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            key="img_conf_slider",
            help="Minimum confidence required to trigger high-priority alerts."
        )
    with col_opt2:
        source_label = st.selectbox(
            "CCTV Surveillance Channel",
            ["Cam 01 - Warehouse Zone B", "Cam 02 - North Corridor", "Cam 03 - Server Room", "Cam 04 - Perimeter Entry", "Manual Operator Upload"],
            key="img_source_select"
        )

    # Analyze Button
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze_clicked = st.button("🔥 ANALYZE CCTV IMAGE", type="primary", use_container_width=True, key="btn_analyze_image")

    if analyze_clicked:
        if not model_status.is_loaded:
            st.error("Cannot perform inference: VGG16 model weights are not loaded.")
            return

        with st.spinner("Executing VGG16 Feature Extraction and Classification..."):
            inference_result = run_vgg16_inference(
                img_pil,
                model_status,
                confidence_threshold=conf_thresh
            )
            time.sleep(0.15)

        if not inference_result["success"]:
            st.error(f"Inference Failure: {inference_result.get('message')}")
            return

        st.session_state["latest_detection"] = {
            "result": inference_result,
            "filename": filename or "cctv_frame.jpg",
            "camera_id": source_label,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # Result Display
    if "latest_detection" in st.session_state:
        det = st.session_state["latest_detection"]
        res = det["result"]
        pred_class = res["predicted_class"]
        conf = res["confidence"]
        conf_pct = res["confidence_percentage"]
        timestamp = det["timestamp"]
        fname = det["filename"]
        cam = det["camera_id"]
        probs = res["all_probabilities"]

        class_lower = pred_class.lower()

        if "fire" in class_lower:
            card_class = "alert-card-fire"
            badge_icon = "🔥"
            badge_title = "CRITICAL ALERT: FIRE DETECTED"
            fill_color = "#EF4444"
        elif "smoke" in class_lower:
            card_class = "alert-card-smoke"
            badge_icon = "⚠️"
            badge_title = "WARNING: SMOKE SIGNATURE DETECTED"
            fill_color = "#F59E0B"
        else:
            card_class = "alert-card-normal"
            badge_icon = "✅"
            badge_title = "STATUS NORMAL: NO FIRE DETECTED"
            fill_color = "#10B981"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="alert-header">
                    <span style="font-size: 2.5rem;">{badge_icon}</span>
                    <div>
                        <h2 class="alert-title">{badge_title}</h2>
                        <div style="font-size: 0.85rem; color: #94A3B8;">
                            Surveillance Feed: <strong>{cam}</strong> | Frame: <strong>{fname}</strong> | Analyzed: <strong>{timestamp}</strong>
                        </div>
                    </div>
                </div>
                
                <div style="margin: 1.25rem 0 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; font-size: 0.9rem; font-weight: 700; margin-bottom: 6px;">
                        <span style="color: #CBD5E1; letter-spacing: 0.02em;">Model Confidence Score</span>
                        <span style="color: {fill_color}; font-size: 1.35rem; font-family: 'JetBrains Mono', monospace; font-weight: 800; text-shadow: 0 0 10px {fill_color}40;">{conf_pct:.2f}%</span>
                    </div>
                    <div style="background: #070B16; border-radius: 9999px; height: 14px; width: 100%; overflow: hidden; border: 1px solid #1E2A47; padding: 2px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
                        <div style="width: {conf_pct}%; height: 100%; border-radius: 9999px; background: linear-gradient(90deg, {fill_color}AA 0%, {fill_color} 100%); box-shadow: 0 0 12px {fill_color}80; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748B; font-family: 'JetBrains Mono', monospace; margin-top: 6px;">
                        <span>Inference Latency: <strong style="color: #94A3B8;">{res['latency_ms']} ms</strong></span>
                        <span>Sensitivity Target: <strong style="color: #94A3B8;">{conf_thresh*100:.0f}%</strong></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_act1, col_act2, col_act3 = st.columns([1.2, 1, 1])
        with col_act1:
            if st.button("💾 Save Result to Detection History", use_container_width=True, type="primary", key="save_img_hist"):
                saved_record = history_manager.add_record(
                    predicted_class=pred_class,
                    confidence=conf,
                    filename=fname,
                    camera_id=cam,
                    notes=f"Latency: {res['latency_ms']}ms",
                    details=res
                )
                st.success(f"Audit record [{saved_record['id']}] saved to detection history!")

        with col_act2:
            json_report = json.dumps({
                "detection_id": f"REP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": timestamp,
                "camera": cam,
                "filename": fname,
                "prediction": pred_class,
                "confidence": conf,
                "probabilities": probs,
                "latency_ms": res["latency_ms"],
                "model": "VGG16 Transfer Learning"
            }, indent=2)
            st.download_button(
                "📥 Export Telemetry JSON",
                data=json_report,
                file_name=f"fireguard_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="down_img_json"
            )

        with col_act3:
            if st.button("🧹 Clear Prediction", use_container_width=True, key="clear_img_pred"):
                del st.session_state["latest_detection"]
                st.rerun()

        with st.expander("🔍 Deep Neural Inspection & Multi-Class Probabilities", expanded=True):
            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                st.markdown("##### Softmax Output Distribution")
                for cname, pval in probs.items():
                    bar_pct = round(pval * 100, 2)
                    st.write(f"**{cname}**: `{bar_pct}%`")
                    st.progress(pval)
            
            with col_p2:
                st.markdown("##### Preprocessed VGG16 Tensor View (224 × 224)")
                resized_preview = img_pil.resize((224, 224))
                st.image(resized_preview, caption="Standard VGG16 Input Tensor (RGB)", width=180)
