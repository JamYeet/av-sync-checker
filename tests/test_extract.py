# tests/test_extract.py
from pathlib import Path
from scipy.io import wavfile

from avsync.extract import extract_audio, extract_frames, crop_faces

SAMPLE_VIDEO = Path("samples/grid/s1/bbaf2n.mpg")


def test_extract_audio_creates_16k_mono_wav(tmp_path):
    out = extract_audio(SAMPLE_VIDEO, out_path=tmp_path / "out.wav")

    assert out.exists()

    # Ensure it has the right sample rate and is mono
    sample_rate, data = wavfile.read(out)
    assert sample_rate == 16000
    assert data.ndim == 1          # mono audio is a 1-D array

def test_extract_frames():
    frames, fps = extract_frames(SAMPLE_VIDEO)

    assert fps == 25
    assert abs(len(frames) - 2.98 * 25) <= 2
    assert frames[0].shape == (288, 360, 3)
    
def test_crop_faces():
    frames, fps = extract_frames(SAMPLE_VIDEO)
    crops = crop_faces(frames, fps)

    # one crop for every frame
    assert len(crops) == len(frames)

    # every crop is 224 x 224 with 3 colour channels
    for crop in crops:
        assert crop.shape == (224, 224, 3)