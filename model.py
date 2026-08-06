import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
MODEL_PATH = "models/best_model.keras"


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