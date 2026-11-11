import subprocess
from pathlib import Path

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