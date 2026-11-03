"""
CCTV Image Upload Component (Manual File Upload)
"""

import streamlit as st
from PIL import Image
from typing import Tuple, Optional
from utils.validators import validate_uploaded_image

def render_upload_selector() -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Renders the direct manual file uploader for CCTV snapshot images.
    
    Returns:
        (PIL.Image or None, filename or None)
    """
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">
                UPLOAD CCTV IMAGE
            </h3>
            <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #94A3B8;">
                Select a high-resolution camera snapshot to identify potential fire or smoke hazards.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Select CCTV snapshot image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png", "webp"],
        key="cctv_uploader",
        help="Maximum supported file size: 25MB"
    )

    chosen_image: Optional[Image.Image] = None
    chosen_name: Optional[str] = None

    if uploaded_file is not None:
        is_valid, err, pil_img = validate_uploaded_image(uploaded_file)
        if not is_valid:
            st.error(f"⚠️ Validation Error: {err}")
        else:
            chosen_image = pil_img
            chosen_name = uploaded_file.name

            col_prev, col_meta = st.columns([1.2, 1])
            with col_prev:
                st.image(chosen_image, caption=f"Active Frame: {chosen_name}", use_container_width=True)
            with col_meta:
                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid #1E2A47; border-radius: 8px; padding: 14px; font-size: 0.82rem;">
                        <div style="color: #38BDF8; font-weight: 700; margin-bottom: 8px;">FRAME SPECIFICATIONS</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #94A3B8;">Source Filename:</span>
                            <strong style="color: #F8FAFC;">{chosen_name}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #94A3B8;">Original Dimensions:</span>
                            <strong style="color: #F8FAFC;">{chosen_image.width} × {chosen_image.height} px</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="color: #94A3B8;">Color Space:</span>
                            <strong style="color: #F8FAFC;">{chosen_image.mode}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">VGG16 Target Res:</span>
                            <strong style="color: #10B981;">224 × 224 px</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    return chosen_image, chosen_name
