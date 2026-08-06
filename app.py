import streamlit as st
import cv2
import os

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
    layout="wide"
)

st.title("🛣️ Road Lane Segmentation using VGG16-UNet")

st.write(
    "Upload an image or a video and the model will segment the road lanes."
)

st.markdown("---")

# --------------------------------------------
# File Upload
# --------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# --------------------------------------------
# If file uploaded
# --------------------------------------------
if uploaded_file is not None:

    extension = uploaded_file.name.split(".")[-1].lower()

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # =====================================================
    # IMAGE
    # =====================================================

    if extension in ["jpg", "jpeg", "png"]:

        with st.spinner("Running lane segmentation..."):

            results = segment_image(file_path)

        original = results["original"]
        mask = results["mask"]
        overlay = results["overlay"]

        mask = (mask * 255).astype("uint8")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original")
            st.image(
                original,
                use_container_width=True
            )

        with col2:
            st.subheader("Predicted Mask")
            st.image(
                mask,
                use_container_width=True
            )

        with col3:
            st.subheader("Overlay")
            st.image(
                overlay,
                use_container_width=True
            )

        # -------------------------------------
        # Download Buttons
        # -------------------------------------

        overlay_bgr = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )

        _, overlay_buffer = cv2.imencode(
            ".png",
            overlay_bgr
        )

        _, mask_buffer = cv2.imencode(
            ".png",
            mask
        )

        st.markdown("---")

        c1, c2 = st.columns(2)

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

    # =====================================================
    # VIDEO
    # =====================================================

    elif extension in ["mp4", "avi", "mov"]:

        st.info("Processing video... Please wait.")

        progress_bar = st.progress(0)

        output_video = os.path.join(
            "outputs",
            "lane_output.mp4"
        )

        process_video(
            file_path,
            output_video,
            progress_bar.progress
        )

        progress_bar.empty()

        st.success("Video processed successfully!")

        st.video(output_video)

        with open(output_video, "rb") as file:

            st.download_button(
                label="⬇ Download Processed Video",
                data=file,
                file_name="lane_segmentation.mp4",
                mime="video/mp4"
            )