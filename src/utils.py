import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import requests
import random

from skimage.util import random_noise
from skimage.transform import warp, rescale as rsc, resize, swirl, PiecewiseAffineTransform
from skimage import io


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_noisy_sinus(num_samples):
    folder_path = create_data_folder()
    data_path = os.path.join(folder_path, 'regression', 'noisy_sinus.npy')
    
    inputs = np.linspace(0.0, 4.0, num_samples, dtype=np.float32)
    ground_truth = np.sin(1.0+inputs*inputs)
    observations = ground_truth + 0.1*np.random.randn(num_samples).astype(np.float32)
    
    data = np.array([inputs, observations, ground_truth])
    np.save(data_path, data)

def get_out_of_distribution_sinus(num_samples):
    inputs = np.linspace(-1.0, 5.0, num_samples, dtype=np.float32)
    ground_truth = np.sin(1.0+inputs*inputs)
    return inputs, ground_truth

def create_house_price():
    folder_path = create_data_folder()
    target_file = os.path.join(folder_path, "regression", "house_prices.csv")
    if not os.path.exists(target_file):
        # download data if not existent yet
        print("Downloading House Pricing Dataset...")
        with requests.Session() as s:
            response = s.get("https://polybox.ethz.ch/index.php/s/CyKkmOuKgsX9b4k/download")
            decoded_content = response.content.decode('utf-8')
        with open(target_file, "w") as f:
            f.write(decoded_content)
        print("...Done")
    
    encoded_x_path = os.path.join(folder_path, "regression", "house_prices_x.npy")
    encoded_y_path = os.path.join(folder_path, "regression", "house_prices_y.npy")
    if not os.path.exists(encoded_x_path):
        df = pd.read_csv(target_file)
        df = df.fillna('')
        print("Encoding Data...")
        unique_values_dict = {}
        for column in df.columns:
            # save unique values of every column in the dictionary
            if column == "Id": continue # ignore id
            unique_values = df[column].unique()
            unique_values_dict[column] = unique_values
            
        def encode_house_price_dataset(row):
            # for every column, give every option an encoding (index of unique values)
            encoded_x = []
            for col in df.columns:
                if col == "Id" or col == "SalePrice": continue # ignore id and SalePrice (y)
                unique_values = unique_values_dict[col]
                value = row[col]
                encoded_x.append(np.where(unique_values == value)[0][0])
            return np.array(encoded_x, dtype=np.int32)
        
        df["encoded_x"] = df.loc[:, df.columns != "SalePrice"].apply(lambda row : encode_house_price_dataset(row), axis=1)
        x_data = np.stack(df["encoded_x"].to_numpy())
        y_data = df["SalePrice"].to_numpy().astype(np.int32)
        
        np.save(encoded_x_path, x_data)
        np.save(encoded_y_path, y_data)
        print("...Done")

def augment(image):
    """Downscaling, flipping, transformation, swirling, change in brightness and contrast
    """
    def aff_trans(image):
        # code adapted from a previous project of Anne: https://github.com/An-nay-marks/3DVision_2022
        # adapted from scikit docs
        rows, cols = image.shape[0], image.shape[1]
        src_cols = np.linspace(0, cols, 5)
        src_rows = np.linspace(0, rows, 2)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        # add sinusoidal oscillation to row coordinates
        dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 2
        dst_cols = src[:, 0]
        dst_rows *= 1.5
        dst_rows -= 1.5 * 7
        dst = np.vstack([dst_cols, dst_rows]).T
        aff = PiecewiseAffineTransform()
        aff.estimate(src, dst)
        return warp(image, aff)    
    
    bw, x, y = image.shape
    image = np.reshape(image,(x, y, bw))
    io.imshow(image)
    plt.show()
    noise = lambda x:random_noise(x)
    swirls = lambda x: swirl(x, strength = 1, radius = 5, rotation=0.05) # not too much of a swirl
    affine_trans = lambda x: aff_trans(x)
    rescale = lambda x: rsc(x, scale=(0.8, 0.8, 1)) # don't change rgb
    
    sampleList = [True, False]
    for func in [rescale, noise, swirls, affine_trans]:
        if random.choice(sampleList):
            image = func(image)
    image = resize(image, (x, y, bw))
    io.imshow(image)
    plt.show()
    return np.reshape(image, (bw, x, y))
    
def create_data_folder():
    folder_path = os.path.join(ROOT_DIR, "data")
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'regression'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'classification'), exist_ok=True)
    return folder_path

def create_model_save_folder(model_type, run_name):
    folder_path = os.path.join(ROOT_DIR, "out", "models", model_type, run_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def load_noisy_sinus():
    data = np.load(os.path.join(os.path.join(ROOT_DIR, "data"), "noisy_sinus.npy"))
    return data

def plot(x, observations, ground_truth=None, outfile=None):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.scatter(x, observations, color="r", label="noisy observation", marker='.')
    if ground_truth is not None: 
        plt.plot(x, ground_truth, color="blue", label="ground truth")
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title("A fairly messy sinus")
    plt.legend(loc="best")
    if outfile is not None: plt.savefig(outfile)
    plt.show()


    