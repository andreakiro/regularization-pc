import torch
import os
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
        
        self.sample_size = 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x = self.X[idx]
        y = self.y[idx]
        return x, y
        

class HousePriceDataset(torch.utils.data.Dataset):
    """
    House Price dataset.

    Each observation is an encoding of different house properties. The groundtruth is the 
    """
    def __init__(
        self,
        data_path: str,
        device: torch.device
    ):
        data_x_path = os.path.join(data_path, "house_prices_x.npy")
        data_y_path = os.path.join(data_path, "house_prices_y.npy")
        X = np.load(data_x_path)
        y = np.load(data_y_path)
        
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
        
        self.sample_size = self.X.shape[1]
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x = self.X[idx]
        y = self.y[idx]
        return x, y