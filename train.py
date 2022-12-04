from torch.utils.data import random_split, DataLoader
from datetime import datetime
import argparse
import torch
import json
import os
import re
import numpy as np

from src.utils import create_noisy_sinus, plot, create_model_save_folder, create_headline_data
from src.mlp.datasets import SinusDataset
from src.transformer.datasets import HeadlineDataset
from src.mlp.trainers import BPTrainer
from src.mlp.models.regression import BPSimpleRegressor, PCSimpleRegressor
from src.transformer.models.transformer import BPTransformer
from src.transformer.trainers import BPTransformerTrainer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, 'out')

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', help=f"Individual run name, if reused the training is resumed", required=True, type=str)
    parser.add_argument('-m', '--model', help=f"Model selection for experiments", choices={'reg', 'clf', 'trf'}, required=True, type=str)
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", choices={'bp', 'pc'}, required=True, type=str)
    parser.add_argument('-n','--num', help=f"Number of samples in the dataset", required=False, default=1000, type=int)
    parser.add_argument('-l','--lr', help=f"Learning rate", required=False, default=0.001, type=float)
    parser.add_argument('-e','--epochs', help=f"Training epochs", required=False, default=300, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-v','--verbose', help=f"Verbosity level", required=False, default=0, type=int)
    parser.add_argument('-i','--init', help=f"PC initialization technique", required=False, default="forward", type=str)
    parser.add_argument('-dp','--dropout', help=f"Dropout level", required=False, default=0, type=float)
    parser.add_argument('-cf','--checkpoint_frequency', help=f"checkpoint frequency in epochs", required=False, default=1, type=int)
    parser.add_argument('-es','--early_stopping', help=f"the number of past epochs taken into account for early_stopping", required=False, default=300, type=int)
    parser.add_argument('-nw','--num_words', help="[Transformer-Only] Amount of words that can be passed to the transformer", required=False, default=8, type=int)
    parser.add_argument('-nh','--num_heads', help="[Transformer-Only] Number of transformer heads", required=False, default=1, type=int)
    parser.add_argument('-el','--enc_layers', help="[Transformer-Only] Number of sub-layers in transformer encoder", required=False, default=3, type=int)
    parser.add_argument('-df','--dim_ffnn', help="[Transformer-Only] Dimension of the feedforward networks in the transformer's encoder layers", required=False, default=2, type=int)
    parser.add_argument('-cp','--cls_pos', help="[Transformer-Only] output position to be used for decoder", required=False, default=0, type=int)
    
    args = vars(parser.parse_args())
    return args


def main():

    args = read_arguments()
    experiment_type = args['model']
    num_samples = args['num']
    lr = args['lr']
    epochs = args['epochs']
    verbose = args['verbose']
    train = args['training']
    arg_plot = args['plot']
    dropout = args['dropout']
    init = args['init']
    run_name = args['run']
    early_stopping = args["early_stopping"]
    checkpoint_frequency = args['checkpoint_frequency']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # saving paths
    model_save_dir = create_model_save_folder(args["model"], run_name)
    log_dir = os.path.join(OUT_DIR, 'logs', args['model'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    image_dir = os.path.join(OUT_DIR, 'images', args['model'], run_name)
    os.makedirs(image_dir, exist_ok=True)
    
    # prepare data and dataloaders
    if experiment_type == "reg":
        data_path = os.path.join(ROOT_DIR, "data/noisy_sinus.npy")
        create_noisy_sinus(num_samples=args['num']) # create the data, if they don't exist
        dataset = SinusDataset(data_path=data_path, device=device)
    
    elif experiment_type == "trf":
        data_path = os.path.join(ROOT_DIR, "data", "headlines_preprocessed.pkl")
        create_headline_data()
        dataset = HeadlineDataset(data_path=data_path, device=device, data_folder_path = os.path.join(ROOT_DIR, "data"))
    
    # create training and validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    training_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(training_data, batch_size=1)
    val_dataloader = DataLoader(val_data, batch_size=1)
    
    # init model and trainer
    if train == "bp" and experiment_type == "reg":
        model = BPSimpleRegressor(
            dropout=dropout
        )
        # TODO Luca mentioned adam is not suitable for PC
        # we might have to change this to SGD if it performs bad on PC
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        loss = torch.nn.MSELoss()
        trainer = BPTrainer(
            optimizer=optimizer, 
            loss=loss,
            checkpoint_frequency=checkpoint_frequency, 
            device=device, 
            epochs=epochs, 
            early_stopping=early_stopping,
            model_save_folder=model_save_dir,
            log_save_folder=log_dir, 
            verbose=verbose)
        
    elif experiment_type == "reg":
        model = PCSimpleRegressor(
            init=init,
            dropout=dropout
        )
    elif train == "bp" and experiment_type == "trf":
        if args['num_words'] > dataset.max_sequence_len:
            raise Exception(f"The parameter 'num_words' cannot be larger than the maximum amount of words, which is {dataset.max_sequence_len}")
        model = BPTransformer(
            d_model = dataset.d_model, 
            max_input_len = args['num_words'], 
            num_heads = args['num_heads'],
            enc_layers = args['enc_layers'], 
            dim_ffnn = args['dim_ffnn'],
            cls_pos = 0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        loss = ...# TODO
        trainer = BPTransformerTrainer(
            optimizer = optimizer,
            loss = loss,
            model_save_folder = model_save_dir,
            log_save_folder = log_dir,
            sequence_len = dataset.max_sequence_len,
            checkpoint_frequency = checkpoint_frequency,
            epochs = epochs,
            early_stopping = early_stopping,
            device = device,
            verbose = verbose
        )
    else:
        raise NotImplementedError
        
    # load last model checkpoint, optimizer checkpoint and past validation losses if existent
    start_epoch = 0
    checkpoints = [checkpoint for checkpoint in os.listdir(model_save_dir) if checkpoint.startswith("checkpoint_")]
    validation_losses = []
    train_losses = []
    if len(checkpoints) != 0:
        last_epoch = max([int(re.search(r'\d+', file).group()) for file in checkpoints])
        start_epoch = last_epoch + 1
        last_checkpoint = torch.load(f = os.path.join(model_save_dir, f"checkpoint_{last_epoch}.pt"))
        model.load_state_dict(last_checkpoint['model_state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])
        validation_losses = np.load(os.path.join(log_dir, f"validation_losses.npy")).tolist()
        train_losses = np.load(os.path.join(log_dir, f"train_losses.npy")).tolist()
    model.to(device)
    
    
    print(f"[Training started]")
    stats = trainer.fit(model, train_dataloader, val_dataloader, start_epoch, val_loss=validation_losses, train_loss=train_losses)
    print(f"\n[Training completed]")
    print(f'{"Number of epochs": <21}: {epochs}')
    print(f'{"Elapsed time": <21}: {round(stats["time"], 2)}s')
    print(f'{"Best train loss": <21}: {round(stats["best_train_loss"], 5)}')
    print(f'{"Best validation loss": <21}: {round(stats["best_val_loss"], 5)}')
    print(f'{"Best epoch": <21}: {stats["best_epoch"]}')

    # evaluate
    X, y = [], []
    for batch, _ in val_dataloader:
        X.append(batch.detach().numpy())
        y.append(model(batch).detach().numpy())
    X, y = np.concatenate(X).ravel(), np.concatenate(y).ravel()
    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")

    # visualize predictions on validation
    if arg_plot:
        outfile = os.path.join(image_dir, dt_string+'.png')
        plot(X, y, dataset.gt, outfile=outfile)
    
    # save model run parameters
    outfile = os.path.join(log_dir, dt_string+'.json')

    log = {
        "framework" : train,
        "epochs" : epochs,
        "optimizer" : type (optimizer).__name__,
        "loss" : loss._get_name(),
        "lr" : lr,
        "results" : stats
    }

    with open(outfile, 'w') as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main()