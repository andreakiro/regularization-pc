from torch.utils.data import random_split, DataLoader
from datetime import datetime
import argparse
import torch
import json
import os
import re
import numpy as np
import random
import numpy as np

from src.utils import create_noisy_sinus, plot, create_model_save_folder
from src.mlp.datasets import SinusDataset
from src.mlp.trainers import BPTrainer, PCTrainer
from src.mlp.models.regression import BPSimpleRegressor, PCSimpleRegressor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, 'out')

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', help=f"Individual run name, if reused the training is resumed", required=True, type=str)
    parser.add_argument('-m', '--model', help=f"Model selection for experiments", choices={'reg', 'clf', 'trf'}, required=True, type=str)
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", choices={'bp', 'pc'}, required=True, type=str)
    parser.add_argument('-n','--num', help=f"Number of generated samples", required=False, default=1000, type=int)
    parser.add_argument('-l','--lr', help=f"Learning rate", required=False, default=0.001, type=float)
    parser.add_argument('-e','--epochs', help=f"Training epochs", required=False, default=300, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-v','--verbose', help=f"Verbosity level", required=False, default=0, type=int)
    parser.add_argument('-i','--init', help=f"PC initialization technique", choices={'zeros', 'normal', 'xavier_normal', 'forward'}, required=False, default="forward", type=str)
    parser.add_argument('-dp','--dropout', help=f"Dropout level", required=False, default=0, type=float)
    parser.add_argument('-cf','--checkpoint_frequency', help=f"checkpoint frequency in epochs", required=False, default=1, type=int)
    parser.add_argument('-es','--early_stopping', help=f"the number of past epochs taken into account for early_stopping", required=False, default=300, type=int)
    parser.add_argument('-b','--batch-size', help=f"Batch size used for training and evaluation", required=False, default=32, type=int)
    parser.add_argument('-lg','--log', help=f"Log info and results of the model or not", required=False, default=False, type=bool)
    args = vars(parser.parse_args())
    return args


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    args = read_arguments()
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
    model_type = args['model']
    batch_size = args['batch_size']
    log = args['log']

    # saving paths
    model_save_dir = create_model_save_folder(args["model"], run_name)
    log_dir = os.path.join(OUT_DIR, 'logs', args['model'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    image_dir = os.path.join(OUT_DIR, 'images', args['model'], run_name)
    os.makedirs(image_dir, exist_ok=True)
    data_path = os.path.join(ROOT_DIR, "src/data/noisy_sinus.npy")
    
    # prepare data and dataloaders
    create_noisy_sinus(num_samples=args['num']) # create the data, if they don't exist
    dataset = SinusDataset(data_path=data_path, device=device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    training_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    if model_type == 'reg':
        if train == 'bp':
            model = BPSimpleRegressor(dropout=dropout).to(device)
        else:
            model = PCSimpleRegressor(dropout=dropout).to(device)
    elif model_type == 'clf':
        raise NotImplementedError("Classifier models are not implemented yet")
    elif model_type == 'trf':
        raise NotImplementedError("Transformer models are not implemented yet")

    # TODO Luca mentioned adam is not suitable for PC
    # we might have to change this to SGD if it performs bad on PC
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
        
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
    
    # init Trainers
    if train == "bp":
        trainer = BPTrainer(
            optimizer=optimizer, 
            loss=loss,
            checkpoint_frequency=checkpoint_frequency, 
            device=device, 
            epochs=epochs, 
            early_stopping=early_stopping,
            val_loss=validation_losses,
            train_loss=train_losses,
            model_save_folder=model_save_dir,
            log_save_folder=log_dir, 
            verbose=verbose)
    elif train == 'pc':
        trainer = PCTrainer(optimizer=optimizer, loss=loss, device=device, init=init, epochs=epochs, verbose=verbose)
    
    print(f"[Training started]")
    stats = trainer.fit(model, train_dataloader, val_dataloader, start_epoch)
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
        plot(X, y, dataset.gt, outfile=outfile if log else None)
    
    # save model run parameters
    if log:
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