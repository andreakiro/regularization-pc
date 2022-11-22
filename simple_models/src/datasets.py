import torch
import numpy as np

class SinusDataset(torch.utils.data.Dataset):
    """
    Noisy Sinus dataset.

    Each observation is an observation of the sinus function in the range [0,4],
    with some random noise applied to it.
    """
    def __init__(
        self,
        data_path: str,
        device: torch.device
    ):

        with open(data_path, 'rb') as f:
            data = np.load(f).astype(np.float32)
            X, y, gt = data

        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
        self.gt = (X, gt)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x = self.X[idx]
        y = self.y[idx]
        return x, y
