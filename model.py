import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
K = tf.keras.backend
load_model = tf.keras.models.load_model

IMG_SIZE = (224, 224)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", PROJECT_DIR / "models" / "best_model.keras")
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Ensure the Git LFS model is "
        "available at models/best_model.keras."
    )


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


model = load_model(
    MODEL_PATH,
    custom_objects={
        "dice_coef": dice_coef,
        "dice_loss": dice_loss
    },
    compile=False
)

print("Model loaded successfully!")
