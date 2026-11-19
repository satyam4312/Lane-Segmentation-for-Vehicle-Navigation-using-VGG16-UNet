"""Download the trained Keras model during deployment when it is not in Git."""

import hashlib
import os
from pathlib import Path
import zipfile

import requests


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_URL = (
    "https://media.githubusercontent.com/media/"
    "satyam4312/Lane-Segmentation-for-Vehicle-Navigation-using-VGG16-UNet/"
    "main/models/best_model.keras"
)
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", PROJECT_DIR / "models" / "best_model.keras")
)
MODEL_URL = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL).strip()
EXPECTED_SHA256 = os.environ.get("MODEL_SHA256", "").strip().lower()


def main():
    if MODEL_PATH.is_file() and zipfile.is_zipfile(MODEL_PATH):
        print(f"Model already exists: {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_PATH.with_name(MODEL_PATH.name + ".download")
    digest = hashlib.sha256()
    downloaded = 0

    try:
        with requests.get(MODEL_URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)

        if downloaded == 0:
            raise RuntimeError("The model download was empty.")

        actual_sha256 = digest.hexdigest()
        if EXPECTED_SHA256 and actual_sha256 != EXPECTED_SHA256:
            raise RuntimeError(
                "Model SHA-256 mismatch: expected "
                f"{EXPECTED_SHA256}, received {actual_sha256}."
            )

        if not zipfile.is_zipfile(temp_path):
            raise RuntimeError(
                "The downloaded file is not a valid .keras archive. Check that "
                "MODEL_URL points directly to the model asset, not a web page."
            )

        temp_path.replace(MODEL_PATH)
        print(
            f"Downloaded model ({downloaded:,} bytes) to {MODEL_PATH}; "
            f"SHA-256: {actual_sha256}"
        )
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
