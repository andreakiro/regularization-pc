import numpy as np
import time
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_noisy_sinus(num_samples):
    folder_path = create_data_folder()
    data_path = os.path.join(folder_path, "noisy_sinus_" + str(int(time.time()))[-4::] + ".npy")
    
    inputs = np.linspace(0.0, 4.0, num_samples, dtype=np.float32)
    ground_truth = np.sin(1.0+inputs*inputs)
    observations = ground_truth + 0.1*np.random.randn(num_samples).astype(np.float32)
    
    data = np.array([inputs, observations, ground_truth])
    np.save(data_path, data)
    
    
def create_data_folder():
    folder_path = os.path.join(ROOT_DIR, "data")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path