import os
import numpy as np
import torch

from utils import ROOT_DIR
from very_simple_regression.bp_ann import BPSimpleRegressor

class bp_trainer:
    def __init__(self, data_file, split=0.8):
        self.network = BPSimpleRegressor()
        data_path = os.path.join(ROOT_DIR, "data", data_file)
        with open(data_path, 'rb') as f:
            data = np.load(f).astype(np.float32)
            print(data)
            x, y, gt = data # y can be used as a noisy observation of the ground truth
            
        # create training and testing data
        training_data_mask = np.random.choice(a=[True, False], size=x.shape, p=[split, 1-split])
        testing_data_mask = np.logical_not(training_data_mask)
        self.train_x = x[training_data_mask]
        self.test_x = x[testing_data_mask]
        self.train_y = y[training_data_mask]
        self.test_y = y[testing_data_mask]
        self.train_gt = gt[training_data_mask]
        self.test_gt = gt[testing_data_mask]
        
    def train():
        ...