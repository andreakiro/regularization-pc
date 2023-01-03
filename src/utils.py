import matplotlib.pyplot as plt
import numpy as np
import os
import random

from skimage.util import random_noise
from skimage.transform import warp, rescale as rsc, resize, swirl, PiecewiseAffineTransform
from tqdm import tqdm
from torchvision import datasets

from .mlp.datasets import OODImageDataset


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def augment_dataset(dataloader):
    """Downscaling, flipping, transformation, swirling, change in brightness and contrast of given 
    MNIST / FashionMNIST dataloader
    returns np arrays for augmented images and groundtruth
    """
    augmented_imgs = np.zeros((len(dataloader), 1, 28, 28)).astype(np.float32)
    groundtruth = np.zeros((len(dataloader)))
    for idx, (img, gt) in enumerate(tqdm(dataloader)):
        augmented_imgs[idx] = augment_single_img(np.squeeze(img.numpy(), axis=0))
        groundtruth[idx] = gt
    
    return  augmented_imgs, groundtruth

def augment_single_img(image):
    bw, x, y = (1, 28, 28)
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
    image = np.reshape(image,(x, y, bw))
    # io.imshow(image)
    # plt.show()
    noise = lambda x:random_noise(x)
    swirls = lambda x: swirl(x, strength = 1, radius = 5, rotation=0.05) # not too much of a swirl
    affine_trans = lambda x: aff_trans(x)
    rescale = lambda x: rsc(x, scale=(0.8, 0.8, 1)) # don't change rgb
    
    sampleList = [True, False]
    for func in [rescale, noise, swirls, affine_trans]:
        if random.choice(sampleList):
            image = func(image)
    image = resize(image, (x, y, bw))
    # io.imshow(image)
    # plt.show()
    image = np.reshape(image, (bw, x, y))
    return image
    
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

def plot_ood_classification(dataset: str):
    if dataset == "mnist":
        data = datasets.MNIST(os.path.join(ROOT_DIR, "data/classification/MNIST"), train=True, download=True).data.numpy()
    elif dataset == "fashion":
        data = datasets.FashionMNIST(os.path.join(ROOT_DIR, "data/classification/FashionMNIST"), train=True, download=True).data.numpy()
    fig, axes = plt.subplots(2, 10, figsize = (9,9))
    for idx in range(10):
        
        img = data[idx]
        img_augmented = augment_single_img(img)
        img = np.array(img, dtype='float')
        pixels = img.reshape((28, 28))
        pixels_augmented = img_augmented.reshape((28, 28))
        axes[0, idx].imshow(pixels, cmap="gray")
        axes[1, idx].imshow(pixels_augmented, cmap = "gray")
        axes[0, idx].get_xaxis().set_visible(False)
        axes[0, idx].get_yaxis().set_visible(False)
        axes[1, idx].get_xaxis().set_visible(False)
        axes[1, idx].get_yaxis().set_visible(False)
    plt.show()

    