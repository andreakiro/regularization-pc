import matplotlib.pyplot as plt
import numpy as np
import os
import random
from skimage.util import random_noise
from skimage.transform import warp, rescale as rsc, resize, swirl, PiecewiseAffineTransform
from torchvision import datasets


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def augment_single_img(image):
    r"""
    Randomly augments the given image with
        - affine transformation, adding sinusoidal oscillation to row columns,
        - gaussian additive noise, 
        - downscaling by a factor of 0.8, and/or
        - swirling the center pixels in a radius of 5

    Parameters
    ----------
        image : np.ndarray 
                image to be augmented, is of shape (1, 28, 28).

    Returns
    -------
        np.ndarray: The augmented image of the same shape as the input.
    """
    bw, x, y = (1, 28, 28)

    def aff_trans(image):
        # code adapted from a previous project of Anne: https://github.com/An-nay-marks/3DVision_2022
        rows, cols = image.shape[0], image.shape[1]
        src_cols = np.linspace(0, cols, 5)
        src_rows = np.linspace(0, rows, 2)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        # add sinusoidal oscillation to row coordinates
        dst_rows = src[:, 1] - \
            np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 2
        dst_cols = src[:, 0]
        dst_rows *= 1.5
        dst_rows -= 1.5 * 7
        dst = np.vstack([dst_cols, dst_rows]).T
        aff = PiecewiseAffineTransform()
        aff.estimate(src, dst)
        return warp(image, aff)

    image = np.reshape(image, (x, y, bw))
    def noise(x): return random_noise(x)
    def swirls(x): return swirl(x, strength=1, radius=5,
                                rotation=0.05)  # not too much of a swirl

    def affine_trans(x): return aff_trans(x)
    def rescale(x): return rsc(x, scale=(0.8, 0.8, 1))  # don't change rgb

    sampleList = [True, False]
    for func in [rescale, noise, swirls, affine_trans]:
        if random.choice(sampleList):
            image = func(image)
    image = resize(image, (x, y, bw))
    image = np.reshape(image, (bw, x, y))
    return image


def create_data_folder():
    r"""
    Creates the data directory structure.

    Returns
    -------
        str: path to the data folder
    """
    folder_path = os.path.join(ROOT_DIR, "data")
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'regression'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'classification'), exist_ok=True)
    return folder_path


def create_model_save_folder(model_type, run_name):
    r"""
    Creates the folder where the model and other logs are saved to.

    Parameters
    ----------
        model_type : str 
                choose between
                "reg" for regression model and 
                "clf" for classification model.
        run_name : str
                The name of the current experiment run.

    Returns
    -------
        str: path to the model saving folder
    """
    folder_path = os.path.join(ROOT_DIR, "out", "models", model_type, run_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def plot(x, observations, ground_truth=None, outfile=None):
    """
    Plots the sinus groundtruth curve as well as the prediction in
    the same figure.

    Parameters
    ----------
        x : np.ndarray
                samples for the x - axis.
        observations : np.ndarray
                predictions of the model corresponding to X
                (same size as x). 
        ground_truth  : np.ndarray
                groundtruth data for the y-axis corresponding to X
                (same size as x). Defaults to None.
        outfile : str (optional)
                Path to the output file, where the plot is saved to. 
                Defaults to None.
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    plt.scatter(x, observations, color="r",
                label="noisy observation", marker='.')
    if ground_truth is not None:
        plt.plot(x, ground_truth, color="blue", label="ground truth")
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title("A fairly messy sinus")
    plt.legend(loc="best")
    if outfile is not None:
        plt.savefig(outfile)
    plt.show()


def plot_ood_classification(dataset: str):
    r"""
    Plots ten samples of the mnist or fashion dataset and plots random augmentation as done
    for the generalization dataset below each sample.

    Parameters
    ----------
        dataset : str
                Choose between "mnist" and "fashion".
    """
    if dataset == "mnist":
        data = datasets.MNIST(os.path.join(
            ROOT_DIR, "data/classification/MNIST"), train=True, download=True).data.numpy()
    elif dataset == "fashion":
        data = datasets.FashionMNIST(os.path.join(
            ROOT_DIR, "data/classification/FashionMNIST"), train=True, download=True).data.numpy()
    fig, axes = plt.subplots(2, 10, figsize=(9, 9))
    for idx in range(10):

        img = data[idx]
        img_augmented = augment_single_img(img)
        img = np.array(img, dtype='float')
        pixels = img.reshape((28, 28))
        pixels_augmented = img_augmented.reshape((28, 28))
        axes[0, idx].imshow(pixels, cmap="gray")
        axes[1, idx].imshow(pixels_augmented, cmap="gray")
        axes[0, idx].get_xaxis().set_visible(False)
        axes[0, idx].get_yaxis().set_visible(False)
        axes[1, idx].get_xaxis().set_visible(False)
        axes[1, idx].get_yaxis().set_visible(False)
    plt.show()
