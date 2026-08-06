import os
import cv2
import numpy as np
from model import model, IMG_SIZE

# -------------------------------
# Create output folder if needed
# -------------------------------
os.makedirs("outputs", exist_ok=True)


# -----------------------------------------
# Read and preprocess image
# -----------------------------------------
def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Convert BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_height, original_width = image.shape[:2]

    resized = cv2.resize(image, IMG_SIZE)

    resized = resized.astype(np.float32) / 255.0

    input_tensor = np.expand_dims(resized, axis=0)

    return (
        image,
        input_tensor,
        original_height,
        original_width,
    )


# -----------------------------------------
# Predict segmentation mask
# -----------------------------------------
def predict_mask(image_tensor):

    prediction = model.predict(image_tensor, verbose=0)

    prediction = prediction[0]

    # Remove channel dimension if present
    if prediction.shape[-1] == 1:
        prediction = np.squeeze(prediction, axis=-1)

    # Binary mask
    mask = (prediction > 0.5).astype(np.uint8)

    return mask


# -----------------------------------------
# Resize mask to original image size
# -----------------------------------------
def resize_mask(mask, width, height):

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask


# -----------------------------------------
# Create colored overlay
# -----------------------------------------
def create_overlay(original_image, mask):

    colored_mask = np.zeros_like(original_image)

    # Green lane mask
    colored_mask[:, :, 1] = mask * 255

    overlay = cv2.addWeighted(
        original_image,
        0.7,
        colored_mask,
        0.3,
        0,
    )

    return colored_mask, overlay


# -----------------------------------------
# Save outputs
# -----------------------------------------
def save_results(mask, overlay):

    cv2.imwrite(
        "outputs/predicted_mask.png",
        mask * 255,
    )

    cv2.imwrite(
        "outputs/overlay.png",
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


# -----------------------------------------
# Complete Prediction Pipeline
# -----------------------------------------
def segment_image(image_path):

    print("Reading image...")

    (
        original_image,
        image_tensor,
        height,
        width,
    ) = preprocess_image(image_path)

    print("Predicting...")

    mask = predict_mask(image_tensor)

    print("Resizing mask...")

    mask = resize_mask(mask, width, height)

    print("Creating overlay...")

    colored_mask, overlay = create_overlay(
        original_image,
        mask,
    )

    print("Saving results...")

    save_results(mask, overlay)

    print("\nPrediction Complete!")
    print("Mask Saved  : outputs/predicted_mask.png")
    print("Overlay Saved: outputs/overlay.png")

    return {
    "original": original_image,
    "mask": mask,
    "overlay": overlay
    }


# -----------------------------------------
# Main Function
# -----------------------------------------
if __name__ == "__main__":

    IMAGE_PATH = "uploads/test.jpg"

    if not os.path.exists(IMAGE_PATH):
        print(f"{IMAGE_PATH} not found.")
    else:
        segment_image(IMAGE_PATH) 