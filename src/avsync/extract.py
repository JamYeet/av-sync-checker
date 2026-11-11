import subprocess
from pathlib import Path

import cv2

def extract_audio(video_path, out_path=None, sample_rate=16000):
    """Extract mono WAV audio from a video using ffmpeg."""
    video_path = Path(video_path)
    if out_path is None:
        out_path = video_path.with_suffix(".wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(out_path)

def extract_frames(video_path):
    """Read all frames from a video as RGB arrays. Returns (frames, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ok, frame = cap.read()          # read the next frame
        if not ok:                  # video finished?
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)              # convert BGR → RGB
        frames.append(frame)

    cap.release()                   # close the video file
    return frames, fps

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

DEFAULT_FACE_MODEL = "models/blaze_face_short_range.tflite"


def crop_faces(frames, fps, model_path=DEFAULT_FACE_MODEL, size=224, margin=0.25):
    """Detect the face in each RGB frame and return square face crops (size x size)."""
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )

    crops = []
    last_box = None

    with vision.FaceDetector.create_from_options(options) as detector:
        for i, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(i * 1000 / fps)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.detections:
                box = result.detections[0].bounding_box
                last_box = (box.origin_x, box.origin_y, box.width, box.height)
            elif last_box is None:
                raise ValueError(f"No face found in frame {i}")

            x, y, w, h = last_box

            # Grow the box by `margin` on each side
            # (e.g. margin=0.25 means 25% of w added to the left AND the right) 
            x1 = int(x - margin * w)
            x2 = int((x + w) + margin * w)
            y1 = int(y - margin * h)
            y2 = int((y + h) + margin * h)
            
            # Clamp x1, y1, x2, y2 to stay inside the image
            # (image height = frame.shape[0], width = frame.shape[1])
            img_h, img_w = frame.shape[0], frame.shape[1]
            
            x1 = max(0, x1)                             # left edge can't go below 0
            y1 = max(0, y1)                             # top edge can't go below 0
            x2 = min(img_w, x2)                         # right edge can't go past the image width
            y2 = min(img_h, y2)                         # bottom edge can't go past the image height
            
            # Slice the face out: frame[y1:y2, x1:x2]
            face = frame[y1:y2, x1:x2]                  # slice: rows (y) first, then columns (x)

            # Resize to (size, size) with cv2.resize and append to crops
            crops.append(cv2.resize(face, (size, size)))
            
    return crops