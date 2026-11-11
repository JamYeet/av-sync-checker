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