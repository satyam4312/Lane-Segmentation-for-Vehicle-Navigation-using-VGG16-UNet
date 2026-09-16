import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import os
import time
import base64
import subprocess
import shutil

from image_predict import segment_image
from video_predict import process_video

# --------------------------------------------
# Create folders
# --------------------------------------------
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------
# Streamlit Configuration
# --------------------------------------------
st.set_page_config(
    page_title="Road Lane Segmentation",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------
# Custom CSS
# --------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }

        .hero {
            background: linear-gradient(120deg, #1f2937 0%, #111827 100%);
            padding: 2.2rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.6rem;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .hero h1 { font-size: 2.1rem; margin-bottom: 0.4rem; color: #ffffff; }
        .hero p { color: #cbd5e1; font-size: 1.02rem; margin: 0; }

        .section-card {
            background: rgba(148, 163, 184, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1.2rem;
        }

        .img-label {
            text-align: center;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 0.5rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }

        div[data-testid="stDownloadButton"] button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
        }

        section[data-testid="stFileUploaderDropzone"] { border-radius: 14px; }

        div[data-testid="stMetric"] {
            background: rgba(148, 163, 184, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.5rem 1.1rem;
            font-weight: 600;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------
# Helpers
# --------------------------------------------
def encode_png_b64(rgb_array):
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (r, g, b)


def build_overlay(original_rgb, mask01, color_rgb, alpha):
    mask_bool = mask01 > 0.5
    color_layer = np.zeros_like(original_rgb)
    color_layer[:, :] = color_rgb
    blended = cv2.addWeighted(original_rgb, 1 - alpha, color_layer, alpha, 0)
    overlay = original_rgb.copy()
    overlay[mask_bool] = blended[mask_bool]
    return overlay


def comparison_slider(before_rgb, after_rgb, height=420):
    before_b64 = encode_png_b64(before_rgb)
    after_b64 = encode_png_b64(after_rgb)
    html = f"""
    <div class="comp-wrap" style="position:relative;width:100%;max-width:100%;
         border-radius:12px;overflow:hidden;user-select:none;">
        <img src="data:image/png;base64,{after_b64}"
             style="width:100%;display:block;">
        <img id="overlayImg" src="data:image/png;base64,{before_b64}"
             style="position:absolute;top:0;left:0;width:100%;height:100%;
             object-fit:cover;clip-path:inset(0 50% 0 0);">
        <div id="handle" style="position:absolute;top:0;left:50%;height:100%;
             width:0;pointer-events:none;">
            <div style="position:absolute;top:0;bottom:0;left:-1px;width:2px;
                 background:white;box-shadow:0 0 6px rgba(0,0,0,0.6);"></div>
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                 width:38px;height:38px;border-radius:50%;background:white;
                 box-shadow:0 2px 8px rgba(0,0,0,0.4);display:flex;align-items:center;
                 justify-content:center;font-size:16px;color:#111;">↔</div>
        </div>
        <input id="slider" type="range" min="0" max="100" value="50"
               style="position:absolute;bottom:10px;left:5%;width:90%;">
        <div style="position:absolute;top:8px;left:10px;background:rgba(0,0,0,0.55);
             color:white;padding:2px 10px;border-radius:6px;font-size:12px;
             font-weight:600;">BEFORE</div>
        <div style="position:absolute;top:8px;right:10px;background:rgba(0,0,0,0.55);
             color:white;padding:2px 10px;border-radius:6px;font-size:12px;
             font-weight:600;">AFTER</div>
    </div>
    <script>
        const slider = document.getElementById('slider');
        const overlayImg = document.getElementById('overlayImg');
        const handle = document.getElementById('handle');
        slider.addEventListener('input', function() {{
            const v = slider.value;
            overlayImg.style.clipPath = 'inset(0 ' + (100 - v) + '% 0 0)';
            handle.style.left = v + '%';
        }});
    </script>
    """
    components.html(html, height=height)


def ensure_browser_playable(video_path):
    """
    OpenCV's VideoWriter commonly outputs mp4v/other codecs that HTML5
    <video> tags can't decode (the player loads but shows 0:00 / no frames).
    Re-encode to H.264 + yuv420p with faststart so it previews inline.
    Falls back to the original file if ffmpeg isn't available or fails.
    """
    if shutil.which("ffmpeg") is None:
        return video_path

    web_path = os.path.splitext(video_path)[0] + "_web.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                web_path
            ],
            check=True,
            capture_output=True
        )
        return web_path
    except Exception:
        return video_path


# --------------------------------------------
# Sidebar
# --------------------------------------------
with st.sidebar:
    st.markdown("### 🛣️ About")
    st.write(
        "This app uses a **VGG16-UNet** segmentation model to detect "
        "and highlight road lanes in images and videos."
    )
    st.markdown("---")
    st.markdown("### 🎛️ Display Controls")
    st.caption("Applies to image results below")
    overlay_color_hex = st.color_picker("Overlay color", "#00FF66")
    overlay_alpha = st.slider("Overlay intensity", 0.0, 1.0, 0.55, 0.05)
    view_mode = st.radio(
        "View mode",
        ["Side by Side", "Slider Compare", "Overlay Only"],
        index=0
    )
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
        1. Pick **Image** or **Video** mode
        2. Upload a file (or use your camera)
        3. Tweak the overlay color / intensity
        4. Compare and download the result
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit • Powered by VGG16-UNet")

# --------------------------------------------
# Hero header
# --------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛣️ Road Lane Segmentation</h1>
        <p>Upload an image or a video and the model will segment the road lanes using a VGG16-UNet architecture.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------
# Mode tabs
# --------------------------------------------
image_tab, video_tab = st.tabs(["📷 Image Mode", "🎥 Video Mode"])

# =====================================================
# IMAGE MODE
# =====================================================
with image_tab:

    source_tab1, source_tab2 = st.tabs(["⬆️ Upload", "📸 Use Camera"])

    uploaded_image = None
    with source_tab1:
        uploaded_image = st.file_uploader(
            "Choose an Image",
            type=["jpg", "jpeg", "png"],
            help="Drag and drop or browse for an image to run lane segmentation on.",
            key="image_uploader"
        )
    with source_tab2:
        camera_image = st.camera_input("Take a photo of a road")
        if camera_image is not None:
            uploaded_image = camera_image

    if uploaded_image is not None:

        file_id = f"{uploaded_image.name}-{uploaded_image.size}" if hasattr(uploaded_image, "name") else "camera-shot"

        if st.session_state.get("img_file_id") != file_id:
            file_path = os.path.join("uploads", getattr(uploaded_image, "name", "camera_capture.jpg"))
            with open(file_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            with st.spinner("Running lane segmentation..."):
                start_time = time.time()
                results = segment_image(file_path)
                elapsed = time.time() - start_time

            mask01 = np.clip(np.squeeze(results["mask"]).astype(float), 0, 1)

            st.session_state["img_file_id"] = file_id
            st.session_state["img_original"] = results["original"]
            st.session_state["img_mask01"] = mask01
            st.session_state["img_elapsed"] = elapsed
            st.toast("Segmentation complete!", icon="✅")

        original = st.session_state["img_original"]
        mask01 = st.session_state["img_mask01"]
        elapsed = st.session_state["img_elapsed"]
        mask_display = (mask01 * 255).astype("uint8")

        overlay_color_rgb = hex_to_bgr(overlay_color_hex)[::-1]  # keep RGB order for RGB arrays
        overlay = build_overlay(original, mask01, overlay_color_rgb, overlay_alpha)

        st.markdown(
            f'<span class="status-pill">✅ Done in {elapsed:.2f}s</span>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Lane Coverage", f"{mask01.mean() * 100:.1f}%")
        m2.metric("Resolution", f"{original.shape[1]}×{original.shape[0]}")
        m3.metric("Inference Time", f"{elapsed:.2f}s")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        if view_mode == "Side by Side":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="img-label">Original</div>', unsafe_allow_html=True)
                st.image(original, use_container_width=True)
            with col2:
                st.markdown('<div class="img-label">Predicted Mask</div>', unsafe_allow_html=True)
                st.image(mask_display, use_container_width=True)
            with col3:
                st.markdown('<div class="img-label">Overlay</div>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

        elif view_mode == "Slider Compare":
            st.markdown('<div class="img-label">Drag to compare Original ↔ Overlay</div>', unsafe_allow_html=True)
            comparison_slider(original, overlay)

        else:  # Overlay Only
            st.markdown('<div class="img-label">Overlay</div>', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🔍 View raw predicted mask"):
            st.image(mask_display, use_container_width=True)

        # -------------------------------------
        # Download Buttons
        # -------------------------------------
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, overlay_buffer = cv2.imencode(".png", overlay_bgr)
        _, mask_buffer = cv2.imencode(".png", mask_display)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇ Download Overlay",
                overlay_buffer.tobytes(),
                "lane_overlay.png",
                "image/png"
            )
        with c2:
            st.download_button(
                "⬇ Download Mask",
                mask_buffer.tobytes(),
                "lane_mask.png",
                "image/png"
            )
        with c3:
            if st.button("🔄 Start Over"):
                for k in ("img_file_id", "img_original", "img_mask01", "img_elapsed"):
                    st.session_state.pop(k, None)
                st.rerun()

    else:
        st.markdown(
            """
            <div class="section-card" style="text-align:center; padding:3rem 1rem;">
                <p style="font-size:1.1rem; color:#94a3b8;">
                    👆 Upload an image or take a photo to get started
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# VIDEO MODE
# =====================================================
with video_tab:

    uploaded_video = st.file_uploader(
        "Choose a Video",
        type=["mp4", "avi", "mov"],
        help="Drag and drop or browse for a video to run lane segmentation on.",
        key="video_uploader"
    )

    playback_col1, playback_col2, playback_col3 = st.columns(3)
    autoplay = playback_col1.checkbox("Autoplay", value=False)
    loop_video = playback_col2.checkbox("Loop", value=False)
    muted = playback_col3.checkbox("Muted", value=True)

    if uploaded_video is not None:

        video_file_id = f"{uploaded_video.name}-{uploaded_video.size}"

        if st.session_state.get("video_file_id") != video_file_id:
            file_path = os.path.join("uploads", uploaded_video.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.info("⏳ Processing video... this may take a moment depending on length.")

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(value):
                progress_bar.progress(value)
                status_text.caption(f"Processing... {int(value * 100)}%")

            output_video = os.path.join("outputs", "lane_output.mp4")

            start_time = time.time()
            process_video(file_path, output_video, update_progress)

            status_text.caption("Optimizing video for playback...")
            playable_video = ensure_browser_playable(output_video)
            elapsed = time.time() - start_time

            progress_bar.empty()
            status_text.empty()
            st.markdown('</div>', unsafe_allow_html=True)

            st.session_state["video_file_id"] = video_file_id
            st.session_state["video_output"] = playable_video
            st.session_state["video_download"] = output_video
            st.session_state["video_elapsed"] = elapsed
            st.balloons()

        output_video = st.session_state["video_output"]
        download_video = st.session_state.get("video_download", output_video)
        elapsed = st.session_state["video_elapsed"]

        st.markdown(
            f'<span class="status-pill">✅ Processed in {elapsed:.1f}s</span>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="img-label">Result Preview</div>', unsafe_allow_html=True)
        with open(output_video, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes, autoplay=autoplay, loop=loop_video, muted=muted)
        st.markdown('</div>', unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            with open(download_video, "rb") as file:
                st.download_button(
                    label="⬇ Download Processed Video",
                    data=file,
                    file_name="lane_segmentation.mp4",
                    mime="video/mp4"
                )
        with d2:
            if st.button("🔄 Process Another Video"):
                for k in ("video_file_id", "video_output", "video_download", "video_elapsed"):
                    st.session_state.pop(k, None)
                st.rerun()

    else:
        st.markdown(
            """
            <div class="section-card" style="text-align:center; padding:3rem 1rem;">
                <p style="font-size:1.1rem; color:#94a3b8;">
                    👆 Upload a video to get started
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )