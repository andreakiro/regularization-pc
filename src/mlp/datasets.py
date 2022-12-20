import os
import torch
import requests
import numpy as np
import pandas as pd
from pathlib import Path


class SinusDataset(torch.utils.data.Dataset):

    @classmethod
    def generate(
        cls,
        data_dir: str,
        num_samples: int,
        lower_bound: float,
        upper_bound: float,
        noise_mean: float = 0,
        noise_std: float = 0.1,
        force_download: bool = False
    ):

        if os.path.exists(data_dir) and not force_download: return data_dir

        x = np.linspace(lower_bound, upper_bound, num_samples, dtype=np.float32)
        gt = np.sin(1.0 + x*x) # sine of some function of inputs x
        y = gt + (noise_std * np.random.randn(num_samples) + noise_mean)

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(data_dir, 'sine_regression_x.npy'), x)
        np.save(os.path.join(data_dir, 'sine_regression_y.npy'), y)
        np.save(os.path.join(data_dir, 'sine_regression_gt.npy'), gt)
        print(f'Created sine dataset at {data_dir}')

        return data_dir

    
    @classmethod
    def out_of_sample(
        cls,
        num_samples: int,
        lower_bound: float,
        upper_bound: float,
        device: torch.device
    ):
        #TODO: torch gives warning: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
        x = torch.tensor(np.linspace(lower_bound, upper_bound, num_samples, dtype=np.float32)).unsqueeze(1).to(device)
        gt = torch.tensor(np.sin(1.0 + x*x), dtype=torch.float32).to(device)
        return x, gt


    def __init__(
        self,
        data_dir: str,
        device: torch.device
    ):
        
        X = np.load(os.path.join(data_dir, 'sine_regression_x.npy'))
        y = np.load(os.path.join(data_dir, 'sine_regression_y.npy'))
        gt = np.load(os.path.join(data_dir, 'sine_regression_gt.npy'))

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

    DOWNLOAD_URL = 'https://polybox.ethz.ch/index.php/s/CyKkmOuKgsX9b4k/download'

    @classmethod
    def generate(cls, data_dir: str, force_download: bool =False):

        if os.path.exists(data_dir) and not force_download: return data_dir

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        target_file = os.path.join(data_dir, 'house_prices_raw.csv')
        
        with requests.Session() as s:
            response = s.get(cls.DOWNLOAD_URL)
            decoded_content = response.content.decode('utf-8')

        with open(target_file, 'w') as f:
            f.write(decoded_content)

        df = pd.read_csv(target_file)
        df = df.fillna('nan_string')

        unique_values_dict = {}
        for column in df.columns:
            # save unique values of every column in the dictionary
            if column == 'Id' or column == 'SalePrice': continue
            unique_values = df[column].unique()
            unique_values_dict[column] = unique_values
        
        
        def encode_house_price_dataset(row):
            encoded_x = []
            for col in df.columns:
                # ignore id and SalePrice (y)
                if col == "Id" or col == "SalePrice": continue 
                unique_values = unique_values_dict[col]
                value = row[col]
                encoded_x.append(np.where(unique_values == value)[0][0])
            return np.array(encoded_x, dtype=np.int32)

        
        df['encoded_x'] = df.apply(lambda row: encode_house_price_dataset(row), axis=1)

        x_features = np.stack(df['encoded_x'].to_numpy())
        y_labels = df['SalePrice'].to_numpy().astype(np.int32)

        np.save(os.path.join(data_dir, 'house_prices_x_features.npy'), x_features)
        np.save(os.path.join(data_dir, 'house_prices_y_labels.npy'), y_labels)
        print(f'Created housing dataset at {data_dir}')

        return data_dir


    def __init__(
        self,
        data_dir: str,
        device: torch.device
    ):
        data_x_path = os.path.join(data_dir, 'house_prices_x_features.npy')
        data_y_path = os.path.join(data_dir, 'house_prices_y_labels.npy')

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

class OODImageDataset(torch.utils.data.Dataset):
    """Out Of Distribution Image Dataset for augmented MNIST or FashionMNIST data
    """
    def __init__(
        self,
        data_dir: str,
        device: torch.device
    ):
        data_x_path = os.path.join(data_dir, 'augmented_images.npy')
        data_y_path = os.path.join(data_dir, 'augmented_images_gt.npy')

        X = np.load(data_x_path)
        y = np.load(data_y_path)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).unsqueeze(1)
        self.device = device
        
    
    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x = self.X[idx].to(self.device)
        y = self.y[idx].to(self.device)
        return x, y