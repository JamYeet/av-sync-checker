import torch
from avsync.sync import load_syncnet


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