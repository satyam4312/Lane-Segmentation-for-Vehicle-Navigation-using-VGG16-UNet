import os
import cv2
import numpy as np
from image_predict import predict_masks, resize_mask, create_overlay

os.makedirs("outputs", exist_ok = True)

# Batch inference reduces the per-call overhead while preserving predictions
# for every source frame. Four frames keeps temporary memory use moderate.
VIDEO_INFERENCE_BATCH_SIZE = 4


def process_video(input_video_path, output_video_path, progress_callback = None):
    """
    Process a video frame by frame and save the output video.
    """

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
    pending_frames = []
    pending_inputs = []

    def write_batch():
        nonlocal frame_number
        masks = predict_masks(np.stack(pending_inputs, axis=0))

        for rgb, mask in zip(pending_frames, masks):
            frame_height, frame_width = rgb.shape[:2]
            mask = resize_mask(mask, frame_width, frame_height)
            _, overlay = create_overlay(rgb, mask)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            frame_number += 1
            if progress_callback is not None and total_frames > 0:
                progress_callback(min(frame_number / total_frames, 1.0))

        pending_frames.clear()
        pending_inputs.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        pending_frames.append(rgb)
        pending_inputs.append(resized)

        if len(pending_inputs) >= VIDEO_INFERENCE_BATCH_SIZE:
            write_batch()

    # Process the final partial batch so no source frames are dropped.
    if pending_inputs:
        write_batch()

    cap.release()
    writer.release()

    return output_video_path

