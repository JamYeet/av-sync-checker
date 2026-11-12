import torch

from avsync.syncnet_model import S

DEFAULT_SYNCNET_WEIGHTS = "models/syncnet_v2.model"


def load_syncnet(weights_path=DEFAULT_SYNCNET_WEIGHTS, device=None):
    """Load pretrained SyncNet in eval mode, on GPU if available."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # "cuda" if a GPU is available, else "cpu"

    model = S()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)        # put the weights into the model
    model = model.to(device)
    model.eval()         # switch to inference mode
    return model