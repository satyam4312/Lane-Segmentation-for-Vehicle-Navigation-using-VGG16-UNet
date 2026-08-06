import os
import cv2
import numpy as np
from image_predict import preprocess_image, predict_mask, resize_mask, create_overlay

os.makedirs("outputs", exist_ok = True)


def process_video(input_video_path, output_video_path, progress_callback=None):
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

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        original_height, original_width = rgb.shape[:2]

        resized = cv2.resize(rgb, (224, 224))

        resized = resized.astype(np.float32) / 255.0

        input_tensor = np.expand_dims(resized, axis=0)

        mask = predict_mask(input_tensor)

        mask = resize_mask(mask, original_width, original_height)

        _, overlay = create_overlay(rgb, mask)

        overlay_bgr = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )

        writer.write(overlay_bgr)

        frame_number += 1

        if progress_callback is not None:
            progress_callback(frame_number / total_frames)

    cap.release()
    writer.release()

    return output_video_path

