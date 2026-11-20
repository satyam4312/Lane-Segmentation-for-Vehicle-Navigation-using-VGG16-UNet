# Road Lane Segmentation

Streamlit app for generating road-lane masks and overlays from images and videos with a TensorFlow VGG16 U-Net model.

![App screenshot](road-lane-segmentation-app.png)

![Processed video example](lane_output_web.gif)

## Features

- Upload JPG, JPEG, or PNG images, take a camera photo, or process MP4, AVI, or MOV video.
- Compare masks and overlays, customize image overlay color, and download results.
- Preview processed videos with FFmpeg.

Inputs are resized to 224 x 224 and normalized to `[0, 1]`. Model predictions above `0.5` are classified as lane pixels. Video is processed frame by frame.

## Model storage

`models/best_model.keras` is stored with **Git LFS**. Git tracks a small pointer; Git LFS stores the 224 MB model file. `.gitattributes` configures LFS for this model. GitHub Free includes 10 GiB of LFS storage and 10 GiB of monthly download bandwidth ([GitHub LFS quotas](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)).

## License

The original project source code is licensed under the MIT License; see [`LICENSE`](LICENSE). The trained model and sample media are excluded because they are derived from or may contain material associated with the TuSimple dataset, whose reuse terms should be confirmed separately before relicensing those artifacts.

## Run locally

Install Git LFS, then from the project directory run:

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, including the Git LFS model.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository and select `app.py` as the entrypoint.
3. Choose Python 3.13 in the app's advanced settings. Community Cloud supports Git LFS files, so the model is available to the app after checkout.

The app loads TensorFlow and processes videos, so deployment may be limited by Community Cloud's available memory and CPU resources.

## Project files

```text
app.py                 Streamlit interface
model.py               Loads the Keras model
image_predict.py       Image preprocessing and prediction
video_predict.py       Frame-by-frame video processing
notebook/              TuSimple training notebook
requirements.txt       Python dependencies
```

The training notebook uses Kaggle TuSimple paths; the dataset is not included.
