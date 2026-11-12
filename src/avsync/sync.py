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
    bgr = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in crops]   # RGB -> BGR (the heads-up from earlier)
    video = np.stack(bgr, axis=0)                     # shape (T, H, W, 3)
    video = video.transpose(3, 0, 1, 2)                   # want (3, T, H, W)
    return torch.from_numpy(video).float().unsqueeze(0)   # add the batch dim -> (1, 3, T, H, W)

def wav_to_mfcc(wav_path):
    """16 kHz mono WAV -> MFCC tensor (1, 1, 13, n_steps), 100 steps per second."""
    sample_rate, audio = wavfile.read(wav_path)
    mfcc = python_speech_features.mfcc(audio, sample_rate)   # shape (n_steps, 13)
    mfcc = mfcc.T                                            # -> (13, n_steps)
    return torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)