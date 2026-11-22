import os
import cv2
import numpy as np
from image_predict import predict_masks, resize_mask, create_overlay

os.makedirs("outputs", exist_ok = True)

# Keep inference tuning internal so the video tab stays simple.
VIDEO_INFERENCE_BATCH_SIZE = 8
VIDEO_INFERENCE_INTERVAL = 1

def process_video(
    input_video_path,
    output_video_path,
    progress_callback=None,
    batch_size=VIDEO_INFERENCE_BATCH_SIZE,
    inference_interval=VIDEO_INFERENCE_INTERVAL,
):
    """
    Process a video and save its lane overlay.

    ``batch_size`` controls how many sampled frames are sent to the model in
    one call. ``inference_interval`` runs inference once every N frames and
    reuses each mask for the frames until the next inference frame.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if inference_interval < 1:
        raise ValueError("inference_interval must be at least 1")

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Unable to open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0
    source_frame_number = 0
    pending_groups = []
    pending_inputs = []
    current_group = []

    def write_batch():
        nonlocal frame_number
        masks = predict_masks(np.stack(pending_inputs, axis=0))

        for group, mask in zip(pending_groups, masks):
            for rgb in group:
                frame_height, frame_width = rgb.shape[:2]
                frame_mask = resize_mask(mask, frame_width, frame_height)
                _, overlay = create_overlay(rgb, frame_mask)
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                frame_number += 1
                if progress_callback is not None and total_frames > 0:
                    progress_callback(min(frame_number / total_frames, 1.0))

        pending_groups.clear()
        pending_inputs.clear()

    def finalize_current_group():
        if not current_group:
            return
        pending_groups.append(current_group.copy())
        resized = cv2.resize(current_group[0], (224, 224)).astype(np.float32) / 255.0
        pending_inputs.append(resized)
        current_group.clear()
        if len(pending_inputs) >= batch_size:
            write_batch()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if source_frame_number % inference_interval == 0:
            finalize_current_group()
        current_group.append(rgb)
        source_frame_number += 1

    # Process the final partial batch so no source frames are dropped.
    finalize_current_group()
    if pending_inputs:
        write_batch()

    cap.release()
    writer.release()

    return output_video_path


