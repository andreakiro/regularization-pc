import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import requests
from torchtext.data.utils import get_tokenizer
from gensim.models import Word2Vec


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
    
    preprocessed_path = os.path.join(folder_path, "headlines_preprocessed.pkl")
    word_2_vec_path = os.path.join(folder_path, "word2vec.model")
    word_2_vec_pair_path = os.path.join(folder_path, "word2vec.wordvectors")
    if not os.path.exists(preprocessed_path):
        # create and save vocabulary if not existent yet
        tokenizer = get_tokenizer('basic_english')
        df = pd.read_csv(target_file)
        
        print("Cleaning Data...")
        def clean(data):
            clean = data.lower()
            clean = re.sub('[^a-zA-Z0-9]', ' ', clean)
            clean = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", clean)
            clean = tokenizer(clean)
            return clean
        df["cleaned_headline"] = df["headline_text"].apply(lambda row : clean(row))
        
        print("Creating Word2Vec Model...")
        model = Word2Vec(sentences=df["cleaned_headline"], vector_size= 100, window = 3, min_count=1, workers=4)
        
        print("Saving Word2Vec Model and Vector Mappings...")
        model.save(word_2_vec_path)
        word_vectors = model.wv
        word_vectors.save(word_2_vec_pair_path)
        
        print("Vectorizing Data with Word2Vec Model...")
        def vectorize(data):
            vec = np.array([model.wv[word] for word in data])
            return vec
        df["vectorized_headline"] = df["cleaned_headline"].apply(lambda row : vectorize(row))
        
        print("Padding Data...")
        df["num_words"] = df["vectorized_headline"].apply(lambda row : len(row))
        max_headline_len = df["num_words"].max()
        def pad(data):
            len_padded = max_headline_len - len(data)
            pad = np.zeros((len_padded if len_padded > 0 else 1, 100), dtype=np.float32) # embedding size 100 for each word that has to be padded
            padded_data = np.concatenate([data, pad])
            return padded_data
        df["padded_vectorized_headline"] = df["vectorized_headline"].apply(lambda row : pad(row))
        
        print("Creating Source (Padding) Masks...")
        def create_mask(vec_headline):
            len_padded = max_headline_len - len(vec_headline)
            mask = np.zeros(max_headline_len, dtype=bool)
            mask[-len_padded::] = True
            return mask
        df["padding_mask"] = df["vectorized_headline"].apply(lambda row: create_mask(row))
        
        print("Saving preprocessed dataframe...")
        df.to_pickle(preprocessed_path)
        print("... Done")
        