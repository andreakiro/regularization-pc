import os
import torch
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils import augment_single_img


class SinusDataset(torch.utils.data.Dataset):
    r"""
    Dataset for value-pairs f(x) = sin(x**2) + u
    where u is iid sampled from a gaussian distribution.

    Parameters:
    ----------
        data_dir : str
                Path to the data directory.
        device : torch.device
                Whether to load the data on cpu or gpu.
        
    """
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
        r"""
        Generates sample pairs in the given interval

        Parameters:
        ----------
            data_dir: str
                    Path to Data Directory.
            num_samples : int
                    Amount of samples bein generated.
            lower_bound : float
                    lower bound of the interval for the x-values
            upper_bound : float
                    upper bound of the interval for the x-values
            noise_mean : float (optional)
                    Mean of the Gaussian to sample the noise for the observation. Defaults to 0.
            noise_std : float (optional)
                    Std of the Gaussian to sample the noise for the observation. Defaults to 0.1.

        Returns:
            data_dir: Path to directory, where the samples are stored in.
        """

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

class OODSinusDataset(torch.utils.data.Dataset):
    r"""
    Dataset for value-pairs f(x) = sin(x**2)
    Used for calculating the generalization error.

    Parameters:
    ----------
        data : (torch.tensor, torch.tensor)
                The (X, y) value pairs previously generated.
        
    """
    @classmethod
    def generate(
        cls,
        num_samples: int,
        lower_bound: float,
        upper_bound: float,
        device: torch.device
    ):
        r"""
        Generates sample pairs in the given interval

        Parameters:
        ----------
            num_samples : int
                    Amount of samples bein generated.
            lower_bound : float
                    lower bound of the interval for the x-values
            upper_bound : float
                    upper bound of the interval for the x-values

        Returns:
        -------
            (torch.Tensor, torch.Tensor): Generated Sample pairs.
        """
        #TODO: torch gives warning: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
        x = torch.tensor(np.linspace(lower_bound, upper_bound, num_samples), dtype=torch.float32).unsqueeze(1).to(device)
        gt = torch.tensor(np.sin(1.0 + x*x), dtype=torch.float32).to(device)
        return (x, gt)


    def __init__(
        self,
        data
    ):
        self.X, self.gt = data
        self.sample_size = 1


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x = self.X[idx]
        gt = self.gt[idx]
        return x, gt



class HousePriceDataset(torch.utils.data.Dataset):
    r"""
    Contains the dataset for the regression of house prices. 
    Dataset taken from
    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

    Parameters:
    ----------
        data_dir : str
                Path to the data directory.
        device : torch.device
                Whether to load the data on cpu or gpu.

    """
    DOWNLOAD_URL = 'https://polybox.ethz.ch/index.php/s/CyKkmOuKgsX9b4k/download'

    @classmethod
    def generate(cls, data_dir: str, force_download: bool =False):
        r"""
        Downloades and Preprocesses the data. The preprocessing contains a
        transformation of the metrics to a numerical value and the normalization 
        of the data.

        Parameters:
        ----------
            data_dir : str
                    Data dir for storing the dataset.
            force_download : bool (optional)
                    Whether to force re-downloading the data, even though the data
                    directory already exists. Defaults to False.

        Returns:
        -------
            str: The data directory, where the data is stored in.
        
        """
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
                encoded_x.append(np.where(unique_values == value)[0][0]/len(unique_values))
            return np.array(encoded_x, dtype=np.float32)

        
        df['encoded_x'] = df.apply(lambda row: encode_house_price_dataset(row), axis=1)

        x_features = np.stack(df['encoded_x'].to_numpy())
        y_labels = df['SalePrice'].to_numpy().astype(np.float32) # y label is the house price
        mean_house_prices = np.mean(y_labels)
        std_house_prices = np.std(y_labels)
        y_labels = (y_labels-mean_house_prices)/std_house_prices

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
    r"""
    Out Of Distribution Image Dataset for containing augmented MNIST or FashionMNIST data
    
    Parameters:
    ----------
        data_dir : str
                Path to the data directory.
        device : torch.device
                Whether to load the data on cpu or gpu.
    
    """
    @classmethod
    def generate(cls, dataloader):
        r"""
        Downscaling, flipping, transformation, swirling, change in brightness and contrast of given 
        MNIST / FashionMNIST dataloader
        
        Parameters:
        ----------
            num_samples : int
                    Amount of samples bein generated.
            lower_bound : float
                    lower bound of the interval for the x-values
            upper_bound : float
                    upper bound of the interval for the x-values

        Returns:
        -------
            (np.ndarray, np.ndarray): Augmented Images and groundtruth pairs.
        """
        augmented_imgs = np.zeros((len(dataloader), 1, 28, 28)).astype(np.float32)
        groundtruth = np.zeros((len(dataloader)))
        for idx, (img, gt) in enumerate(tqdm(dataloader)):
            augmented_imgs[idx] = augment_single_img(np.squeeze(img.numpy(), axis=0))
            groundtruth[idx] = gt
        
        return augmented_imgs, groundtruth

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