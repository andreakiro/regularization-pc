import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import requests
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_noisy_sinus(num_samples):
    folder_path = create_data_folder()
    data_path = os.path.join(folder_path, "noisy_sinus.npy")
    
    inputs = np.linspace(0.0, 4.0, num_samples, dtype=np.float32)
    ground_truth = np.sin(1.0+inputs*inputs)
    observations = ground_truth + 0.1*np.random.randn(num_samples).astype(np.float32)
    
    data = np.array([inputs, observations, ground_truth])
    np.save(data_path, data)
    
def create_data_folder():
    folder_path = os.path.join(ROOT_DIR, "data")
    os.makedirs(folder_path, exist_ok=True)
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
        plt.plot(ground_truth[0], ground_truth[1], color="blue", label="ground truth")
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title("A fairly messy sinus")
    plt.legend(loc="best")
    if outfile is not None: plt.savefig(outfile)
    plt.show()

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def create_headline_data():
    # dataset for transformer trained for natural language generation
    folder_path = create_data_folder()
    target_file = os.path.join(folder_path, "headlines.csv")
    if not os.path.exists(target_file):
        # download data if not existent yet
        print("Downloading dataset...")
        with requests.Session() as s:
            response = s.get("https://drive.google.com/u/0/uc?id=1sKEXpxbw8Xipz2QFUz0BvnHjIYLf2Rhf&export=download")
            decoded_content = response.content.decode('utf-8')
        with open(target_file, "w") as f:
            f.write(decoded_content)        
        print("...Done")
    
    vocab_path = os.path.join(folder_path, "headlines_vocabulary.pth")
    if not os.path.exists(vocab_path):    
        # create and save vocabulary if not existent yet
        print("Creating Vocabulary...")
        tokenizer = get_tokenizer('basic_english')
        df = pd.read_csv(target_file)
        headlines_to_one_text = df['headline_text'].agg(lambda x: ' '.join(x.dropna())).split(' ')
        vocab = build_vocab_from_iterator(map(tokenizer, headlines_to_one_text), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        torch.save(vocab, vocab_path)
        print("... Done")