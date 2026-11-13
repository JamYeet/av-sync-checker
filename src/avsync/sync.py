import torch
import cv2
import numpy as np
import python_speech_features
from scipy.io import wavfile
from avsync.syncnet_model import S



DEFAULT_SYNCNET_WEIGHTS = "models/syncnet_v2.model"


def load_syncnet(weights_path=DEFAULT_SYNCNET_WEIGHTS, device=None):
    """Load pretrained SyncNet in eval mode, on GPU if available."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = S()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model

def crops_to_tensor(crops):
    """List of RGB face crops -> float tensor (1, 3, T, H, W) in BGR, as SyncNet expects."""
    bgr = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in crops]   # RGB -> BGR
    video = np.stack(bgr, axis=0)                     # shape (T, H, W, 3)
    video = video.transpose(3, 0, 1, 2)                   # want (3, T, H, W)
    return torch.from_numpy(video).float().unsqueeze(0)   # add the batch dim -> (1, 3, T, H, W)

def wav_to_mfcc(wav_path):
    """16 kHz mono WAV -> MFCC tensor (1, 1, 13, n_steps), 100 steps per second."""
    sample_rate, audio = wavfile.read(wav_path)
    mfcc = python_speech_features.mfcc(audio, sample_rate)   # shape (n_steps, 13)
    mfcc = mfcc.T                                            # -> (13, n_steps)
    return torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)

def embed_windows(model, video, mfcc, batch_size=64):
    """Slide a 5-frame / 20-step window along the clip.
    Returns (lip_embs, aud_embs), each of shape (N, 1024)."""
    device = next(model.parameters()).device
    min_length = min(video.shape[2], mfcc.shape[3] // 4)   # frames we have both video AND audio for
    n_windows = min_length - 4                               # last window must still fit 5 frames

    lip_embs, aud_embs = [], []
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            idx = range(start, min(start + batch_size, n_windows))
            lip_batch = torch.cat([video[:, :, i:i+5] for i in idx])     # 5 frames from i
            aud_batch = torch.cat([mfcc[:, :, :, 4*i:4*i+20] for i in idx])   # 20 audio steps from 4*i
            lip_embs.append(model.forward_lip(lip_batch.to(device)).cpu())
            aud_embs.append(model.forward_aud(aud_batch.to(device)).cpu())

    return torch.cat(lip_embs), torch.cat(aud_embs)

def find_offset(lip_embs, aud_embs, max_shift=15, fps=25):
    """Try audio shifts of -max_shift..+max_shift frames; return the best one.
    Positive offset = audio lags the video."""
    n = lip_embs.shape[0]
    shifts = list(range(-max_shift, max_shift + 1))
    mean_dists = []

    for s in shifts:
        if s >= 0:
            lip, aud = lip_embs[:n - s], aud_embs[s:]      # lip[i] vs aud[i+s]
        else:
            lip, aud = lip_embs[-s:], aud_embs[:n + s]     # lip[i] vs aud[i-|s|]
        d = torch.nn.functional.pairwise_distance(lip, aud)   # one distance per window
        mean_dists.append(d.mean())

    mean_dists = torch.stack(mean_dists)        # one average distance per shift
    best = torch.argmin(mean_dists).item()                            # INDEX of the smallest distance
    offset = shifts[best]
    confidence = (mean_dists.median() - mean_dists[best]).item()

    return {
        "offset_frames": offset,
        "offset_ms": 1000/fps * offset,                    # frames → milliseconds
        "confidence": confidence,
        "mean_dists": mean_dists,
    }