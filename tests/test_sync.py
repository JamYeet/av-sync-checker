from pathlib import Path

import torch

from scipy.io import wavfile

from avsync.extract import extract_audio, extract_frames, crop_faces
from avsync.sync import load_syncnet, crops_to_tensor, wav_to_mfcc, embed_windows, find_offset, shift_audio, audio_to_mfcc

SAMPLE_VIDEO = Path("samples/grid/s1/bbaf2n.mpg")

def test_syncnet_loads_and_runs():
    model = load_syncnet()
    device = next(model.parameters()).device

    lips = torch.zeros(1, 3, 5, 224, 224, device=device)   # 1 clip, 3 colours, 5 frames, 224x224
    audio = torch.zeros(1, 1, 13, 20, device=device)       # 1 clip, 13 MFCCs, 20 audio steps (0.2 s)

    with torch.no_grad():
        lip_emb = model.forward_lip(lips)
        aud_emb = model.forward_aud(audio)

    # assert both outputs have shape (1, 1024)
    assert lip_emb.shape == (1, 1024)
    assert aud_emb.shape == (1, 1024)
    
    
def test_prepare_inputs(tmp_path):  
    # video side
    frames, fps = extract_frames(SAMPLE_VIDEO)
    crops = crop_faces(frames, fps)
    video = crops_to_tensor(crops)

    # audio side
    wav = extract_audio(SAMPLE_VIDEO, out_path=tmp_path / "a.wav")
    mfcc = wav_to_mfcc(wav)
    
    assert video.shape == (1, 3, len(crops), 224, 224)
    assert mfcc.shape[2] == 13
    assert abs(mfcc.shape[3] - len(crops) * 4) <= 4
    
def test_embed_windows(tmp_path):
    frames, fps = extract_frames(SAMPLE_VIDEO)
    crops = crop_faces(frames, fps)
    video = crops_to_tensor(crops)
    
    wav = extract_audio(SAMPLE_VIDEO, out_path=tmp_path / "a.wav")
    mfcc = wav_to_mfcc(wav)
    
    model = load_syncnet()
    
    lip_embs, aud_embs = embed_windows(model, video, mfcc)
    
    assert lip_embs.shape[0] == aud_embs.shape[0]

    assert lip_embs.shape[1] == 1024
    assert aud_embs.shape[1] == 1024

    expected = min(video.shape[2], mfcc.shape[3] // 4) - 4
    assert lip_embs.shape[0] == expected
    
def test_find_offset_in_sync_clip(tmp_path):
    frames, fps = extract_frames(SAMPLE_VIDEO)
    crops = crop_faces(frames, fps)
    video = crops_to_tensor(crops)

    wav = extract_audio(SAMPLE_VIDEO, out_path=tmp_path / "a.wav")
    mfcc = wav_to_mfcc(wav)

    model = load_syncnet()
    lip_embs, aud_embs = embed_windows(model, video, mfcc)

    result = find_offset(lip_embs, aud_embs)
    print("offset:", result["offset_frames"], "frames,", result["offset_ms"], "ms | confidence:", result["confidence"])

    assert abs(result["offset_frames"]) <= 1            # offset within 1 frame of 0
    assert result["confidence"] > 0                     # confidence is positive

def test_detects_known_audio_shift(tmp_path):
    frames, fps = extract_frames(SAMPLE_VIDEO)
    video = crops_to_tensor(crop_faces(frames, fps))

    wav = extract_audio(SAMPLE_VIDEO, out_path=tmp_path / "a.wav")
    sample_rate, audio = wavfile.read(wav)

    model = load_syncnet()

    for offset_ms in [80, -120]:
        shifted = shift_audio(audio, sample_rate, offset_ms)
        mfcc = audio_to_mfcc(shifted, sample_rate)
        lip_embs, aud_embs = embed_windows(model, video, mfcc)
        result = find_offset(lip_embs, aud_embs)

        print(offset_ms, "ms shifted →", result["offset_ms"], "ms detected | confidence", result["confidence"])

        assert abs(result["offset_ms"] - offset_ms) <= 40