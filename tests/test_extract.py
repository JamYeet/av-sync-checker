# tests/test_extract.py
from pathlib import Path
from scipy.io import wavfile

from avsync.extract import extract_audio, extract_frames

SAMPLE_VIDEO = Path("samples/grid/s1/bbaf2n.mpg")


def test_extract_audio_creates_16k_mono_wav(tmp_path):
    out = extract_audio(SAMPLE_VIDEO, out_path=tmp_path / "out.wav")

    assert out.exists()

    # 2. Ensure it has the right sample rate and is mono
    sample_rate, data = wavfile.read(out)
    assert sample_rate == 16000
    assert data.ndim == 1          # mono audio is a 1-D array

def test_extract_frames():
    frames, fps = extract_frames(SAMPLE_VIDEO)

    assert fps == 25
    assert abs(len(frames) - 2.98 * 25) <= 2
    assert frames[0].shape == (288, 360, 3)