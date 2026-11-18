"""Download the model files on first use, so the repo works on a fresh machine."""
import urllib.request
from pathlib import Path

MODELS = {
    "models/syncnet_v2.model":
        "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
    "models/blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}


def ensure_models():
    """Download any model file that isn't already on disk."""
    for path, url in MODELS.items():
        path = Path(path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {path.name}...")
            urllib.request.urlretrieve(url, path)