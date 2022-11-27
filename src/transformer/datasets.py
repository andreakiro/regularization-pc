import torch
import numpy as np
import pandas as pd
from torchtext.data.utils import get_tokenizer
from utils import ROOT_DIR
import os

class HeadlineDataset(torch.utils.data.Dataset):
    """
    Headline dataset for Natural Language Generation.

    Each observation is the beginning of the headline
    with some random noise applied to it.
    """
    def __init__(
        self,
        data_path: str,
        device: torch.device
    ):

        with open(data_path, 'r') as target_file:
            df = pd.read_csv(target_file)
        tokenizer = get_tokenizer('basic_english')
        vocab = torch.load(os.path.join(ROOT_DIR, "data", "headlines_vocabulary.pth"))
        vocab.set_default_index(vocab['<unk>'])
        
        def data_process(raw_text_iter):
            """Converts raw text into a flat Tensor"""
            data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
        
        self.data = data_process(df["headline_text"]).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x = self.data[idx]
        return x
