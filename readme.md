# Road Lane Segmentation

**See lane markings highlighted in a photo or video.** Upload a road image, take a camera photo, or process a short clip with a VGG16 U-Net model.

<p align="center">
  <a href="https://lane-segmentation-app.streamlit.app/"><strong>🚘 Open the live app</strong></a>
</p>

![Road Lane Segmentation app](road-lane-segmentation-app.png)

## What you can do

- Segment JPG, JPEG, and PNG images, or capture a photo with your camera.
- Compare the original image and predicted mask side by side, with a comparison slider, or as a color overlay.
- Change the overlay color and intensity, then download the mask or overlay.
- Process MP4, AVI, and MOV videos frame by frame, preview the result, and download it.

![Example processed video](lane_output_web.gif)

## How it works

The app resizes each frame to 224 × 224 pixels, scales RGB values to `[0, 1]`, and runs the model. Predictions above `0.5` become lane pixels. The mask is resized to the input dimensions and blended with the original image. Video uses the same process one frame at a time.

## Where lane segmentation can help

Lane-marking segmentation turns road images into a visual mask that is easier to inspect and use in computer-vision experiments. Similar tools can support:

- **Driver-assistance and autonomous-driving research:** prototype lane-awareness pipelines and review model predictions on road scenes.
- **Fleet and logistics teams:** analyze sampled dashcam footage to understand lane visibility along routes.
- **Smart-city and road-maintenance teams:** explore camera-based surveys for finding locations where lane markings may need inspection or repainting.
- **Education and prototyping:** teach image segmentation and compare model outputs without building a full perception stack.

These are potential applications of the technique. This project is a visual demo; it does not control vehicles, certify road conditions, or replace safety-critical validation.

## Future directions

Possible next steps include evaluating on more regions and weather conditions, improving consistency across adjacent video frames, measuring accuracy with standard segmentation metrics, and optimizing inference for edge devices. A batch workflow or API could also help teams integrate predictions into larger road-review tools.

## Run locally

Python 3.13 and Git LFS are recommended. From a terminal:

```bash
git clone https://github.com/satyam4312/Lane-Segmentation-for-Vehicle-Navigation-using-VGG16-UNet.git
cd Lane-Segmentation-for-Vehicle-Navigation-using-VGG16-UNet
git lfs install
git lfs pull
python -m venv .venv
```

Activate the environment, install dependencies, and launch the app:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell: .\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
streamlit run app.py
```

## Model

The Keras model is stored at `models/best_model.keras` with Git LFS (about 224 MB). Make sure Git LFS is installed and the model has been downloaded before running the app. The training notebook uses TuSimple dataset paths; the dataset itself is not included.

## Deploy

This app is deployed on [Streamlit Community Cloud](https://share.streamlit.io/) from `app.py`. The public demo is available at [lane-segmentation-app.streamlit.app](https://lane-segmentation-app.streamlit.app/). Community Cloud supports Git LFS, but TensorFlow inference and video processing can be constrained by its shared CPU and memory resources.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and image/video workflow |
| `model.py` | Loads the Keras segmentation model |
| `image_predict.py` | Image preprocessing, prediction, and overlay helpers |
| `video_predict.py` | Frame-by-frame video processing |
| `notebook/` | TuSimple training notebook |
| `models/best_model.keras` | Model weights, stored with Git LFS |

## License

The original project source code is licensed under the [MIT License](LICENSE). The model weights and sample media are excluded from that license because their reuse terms may depend on the TuSimple dataset. Check the dataset terms before redistributing or relicensing those files.
