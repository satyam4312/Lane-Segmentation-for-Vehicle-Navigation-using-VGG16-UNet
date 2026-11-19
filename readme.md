# Road Lane Segmentation

Streamlit app for creating lane masks and overlays from road images and videos using a TensorFlow VGG16 U-Net model.

![App screenshot](road-lane-segmentation-app.png)

![Processed video example](lane_output_web.gif)

## Features

- Upload JPG, JPEG, or PNG images, take a camera photo, or upload MP4, AVI, or MOV video.
- Compare image masks and overlays; adjust overlay color and intensity.
- Download image masks, overlays, and processed videos.
- Preview processed videos with the bundled FFmpeg encoder.

The model resizes each image or video frame to 224 x 224, normalizes pixels to `[0, 1]`, and labels predictions above `0.5` as lane pixels. Video is processed one frame at a time.

## Run locally

The model file is required at `models/best_model.keras`. It is excluded from Git. If you have a direct download URL, set `MODEL_URL` and run `python download_model.py` before starting the app; otherwise, place the file in `models/` yourself.

PowerShell download example:

```powershell
$env:MODEL_URL = "https://github.com/OWNER/REPOSITORY/releases/download/TAG/best_model.keras"
python download_model.py
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell: replace the activation command with `\.venv\Scripts\Activate.ps1`.

Run the commands from the project directory. The app prints its local URL when ready and creates `uploads/` and `outputs/` as needed.

## Deploy on Render

The model is about 224 MB and is intentionally excluded by `.gitignore`, so it is not committed to the repository. The local file can still be attached directly to a GitHub Release; release assets are separate from tracked Git files. Render downloads the release asset during its build.

1. In GitHub, create a public release and attach the local `models/best_model.keras` file under **Attach binaries**. Do not add or commit it to the repository. Release assets support files up to 2 GiB ([GitHub limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)).
2. Copy the asset's direct download URL. It should look like:

   ```text
   https://github.com/OWNER/REPOSITORY/releases/download/TAG/best_model.keras
   ```

3. In Render, create a **Blueprint** from this repository. When prompted, set:
   - `MODEL_URL`: the direct release-asset URL
   - `MODEL_SHA256`: optional checksum for download verification
4. Deploy. `render.yaml` installs dependencies, downloads the model and bundled FFmpeg, then starts Streamlit.

The Blueprint uses Render's Standard plan (1 CPU, 2 GB RAM). The free plan has 512 MB RAM and is likely too small for TensorFlow and this model. Check [current Render pricing](https://render.com/pricing). The model is included in each build, so no persistent disk is needed; a new build downloads it again.

To calculate the optional checksum on Windows, run:

```powershell
Get-FileHash .\models\best_model.keras -Algorithm SHA256
```

## Deployment checks

- Render needs `MODEL_URL` set before the first build. Use the public release asset's direct download URL, not the release page URL.
- The build validates the downloaded model and optional SHA-256 checksum, then fails with a clear error if either is invalid.
- The Blueprint selects a 2 GB RAM plan because TensorFlow may exceed the free plan's 512 MB limit.
- FFmpeg installs with `requirements.txt` and is used automatically for video previews.

## Project files

```text
app.py                 Streamlit interface
model.py               Loads the Keras model
download_model.py      Downloads the model during Render builds
image_predict.py       Image preprocessing and prediction
video_predict.py       Frame-by-frame video processing
render.yaml            Render Blueprint configuration
notebook/              TuSimple training notebook
requirements.txt       Python dependencies
```

The notebook uses TuSimple data paths from Kaggle. The dataset and training files are not included; the app only needs the trained model file.
