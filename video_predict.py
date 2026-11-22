import os
import math
import cv2
import numpy as np
from image_predict import predict_mask, resize_mask, create_overlay

os.makedirs("outputs", exist_ok = True)

# Run the model at a lower rate than the source video to reduce CPU use on
# Streamlit Community Cloud. The output still keeps every source frame and
# its original playback rate; skipped frames reuse the latest lane mask.
VIDEO_INFERENCE_FPS = float(os.environ.get("VIDEO_INFERENCE_FPS", "10"))


def process_video(input_video_path, output_video_path, progress_callback=None):
    """
    Process a video frame by frame and save the output video.
    """

    cap = cv2.VideoCapture(input_video_path)
    writer = None

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Unable to open this video. Try exporting it as MP4 (H.264).")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read the video's frame dimensions.")
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        if not math.isfinite(VIDEO_INFERENCE_FPS) or VIDEO_INFERENCE_FPS <= 0:
            raise RuntimeError("VIDEO_INFERENCE_FPS must be a positive number.")

        # Infer at most 10 frames per second by default, reusing the most
        # recent mask for intervening frames while retaining source timing.
        inference_interval = max(1, math.ceil(fps / VIDEO_INFERENCE_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create the output video. Check available disk space and video dimensions.")

        frame_number = 0
        mask = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if mask is None or frame_number % inference_interval == 0:
                resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
                input_tensor = np.expand_dims(resized, axis=0)
                predicted_mask = predict_mask(input_tensor)
                mask = resize_mask(predicted_mask, width, height)

            _, overlay = create_overlay(rgb, mask)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            frame_number += 1
            if progress_callback is not None and total_frames > 0:
                progress_callback(min(frame_number / total_frames, 1.0))

        if frame_number == 0:
            raise RuntimeError("No readable frames were found in this video.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not os.path.isfile(output_video_path) or os.path.getsize(output_video_path) == 0:
        raise RuntimeError("Video processing finished, but no output video was created.")

    return output_video_path

