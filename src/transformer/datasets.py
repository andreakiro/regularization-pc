import torch
import pandas as pd
import numpy as np

class HeadlineDataset(torch.utils.data.Dataset):
    """
    Headline dataset for Natural Language Generation.

    Each observation is the beginning of the headline
    with some random noise applied to it.
    """
    def __init__(
        self,
        data_path: str,
        data_folder_path: str,
        device: torch.device
    ):
        print("Retrieving data for training...")
        with open(data_path, 'rb') as target_file:
            df = pd.read_pickle(target_file)
        self.data_folder_path = data_folder_path
        self.device = device
        self.d_model = 100
        self.max_sequence_len = 15
        self.data = df["vectorized_headline"]
        print("...Done")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indices):
        if torch.is_tensor(indices): indices = indices.tolist()
        if isinstance(indices, int): 
            indices = [indices]
        return torch.Tensor(np.concatenate(self.data[self.data.index.isin(indices)].values, axis=0)).to(self.device)
