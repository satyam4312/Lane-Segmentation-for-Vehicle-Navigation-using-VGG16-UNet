"""Fetch and verify the public model object from GitHub LFS."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOSITORY = "satyam4312/Lane-Segmentation-for-Vehicle-Navigation-using-VGG16-UNet"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.keras"
LFS_OID = "aeb3cc788950c2611c8824e38bf42545c7fa83d7d0abbfed373ae397dcc92a58"
MODEL_SIZE = 223_816_349
LFS_BATCH_URL = f"https://github.com/{REPOSITORY}.git/info/lfs/objects/batch"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_download_action():
    payload = {
        "operation": "download",
        "transfers": ["basic"],
        "ref": {"name": "refs/heads/main"},
        "objects": [{"oid": LFS_OID, "size": MODEL_SIZE}],
    }
    request = Request(
        LFS_BATCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "lane-segmentation-model-downloader",
        },
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    objects = result.get("objects", [])
    if not objects:
        raise RuntimeError(f"Git LFS returned no model object: {result}")

    model_object = objects[0]
    if "error" in model_object:
        raise RuntimeError(f"Git LFS model request failed: {model_object['error']}")

    action = model_object.get("actions", {}).get("download")
    if not action or not action.get("href"):
        raise RuntimeError("Git LFS did not return a model download URL.")
    return action


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size == MODEL_SIZE:
        if sha256_file(MODEL_PATH) == LFS_OID:
            print("Git LFS model is already present and verified.")
            return

    action = get_download_action()
    request = Request(
        action["href"],
        headers=action.get("header", {}),
        method="GET",
    )

    temp_path = None
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with urlopen(request, timeout=180) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=MODEL_PATH.parent, delete=False
            ) as output:
                temp_path = Path(output.name)
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)

        actual_oid = digest.hexdigest()
        if downloaded != MODEL_SIZE:
            raise RuntimeError(
                f"Model download size mismatch: expected {MODEL_SIZE}, got {downloaded}."
            )
        if actual_oid != LFS_OID:
            raise RuntimeError(
                f"Model SHA-256 mismatch: expected {LFS_OID}, got {actual_oid}."
            )

        os.replace(temp_path, MODEL_PATH)
        temp_path = None
        print(f"Downloaded and verified Git LFS model ({downloaded:,} bytes).")
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not download the model from GitHub LFS: {error}") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    download_model()
