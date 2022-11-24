from torch.utils.data import random_split, DataLoader
from datetime import datetime
import argparse
import torch
import json
import os
import re
import numpy as np

from src.utils import create_noisy_sinus, plot, create_model_save_folder
from src.mlp.datasets import SinusDataset
from src.mlp.trainers import BPTrainer
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
    parser.add_argument('-i','--init', help=f"PC initialization technique", required=False, default="forward", type=str)
    parser.add_argument('-dp','--dropout', help=f"Dropout level", required=False, default=0, type=float)
    parser.add_argument('-cf','--checkpoint_frequency', help=f"checkpoint frequency in epochs", required=False, default=1, type=int)
    
    args = vars(parser.parse_args())
    return args


def main():

    args = read_arguments()

    create_noisy_sinus(num_samples=args['num']) # create the data, if they don't exist

    lr = args['lr']
    epochs = args['epochs']
    verbose = args['verbose']
    train = args['training']
    arg_plot = args['plot']
    out_dir = args['output_dir']
    dropout = args['dropout']
    init = args['init']
    run_name = args['run']
    checkpoint_frequency = args['checkpoint_frequency']
    
    model_save_folder = create_model_save_folder(args["model"], run_name)

    data_path = os.path.join(ROOT_DIR, "src/data/noisy_sinus.npy")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = SinusDataset(data_path=data_path, device=device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    training_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(training_data, batch_size=32)
    val_dataloader = DataLoader(val_data, batch_size=32)

    if train == "bp":
        model = BPSimpleRegressor(
            dropout=dropout
        )

    else:
        model = PCSimpleRegressor(
            init=init,
            dropout=dropout
        )
        
    # load last model checkpoint, optimizer checkpoint and past validation losses if existent
    start_epoch = 0
    # model_archive = [model_file for model_file in os.listdir(model_save_folder) if model_file.startswith("model_")]
    # optimizer_archive = [optimizer_file for optimizer_file in os.listdir(model_save_folder) if optimizer_file.startswith("optimizer_")]
    checkpoints = [checkpoint for checkpoint in os.listdir(model_save_folder) if checkpoint.startswith("checkpoint_")]
    validation_losses = []
    train_losses = []
    
    # TODO Luca mentioned adam is not suitable for PC
    # we might have to change this to SGD if it performs bad on PC
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    loss = torch.nn.MSELoss()
    
    if len(checkpoints) != 0:
        last_epoch = max([int(re.search(r'\d+', file).group()) for file in checkpoints])
        start_epoch = last_epoch + 1
        last_checkpoint = torch.load(f = os.path.join(model_save_folder, f"checkpoint_{last_epoch}.pt"))
        model.load_state_dict(last_checkpoint['model_state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])
        # model = torch.load(f=os.path.join(model_save_folder, f"model_ep{last_epoch}"), map_location=model)
        validation_losses = np.load(os.path.join(model_save_folder, f"validation_losses.npy")).tolist()
        train_losses = np.load(os.path.join(model_save_folder, f"train_losses.npy")).tolist()
    model.to(device)

    print(f"[Training started]")

    if train == "bp":
        trainer = BPTrainer(
            optimizer=optimizer, 
            loss=loss,
            checkpoint_frequency=checkpoint_frequency, 
            device=device, 
            epochs=epochs, 
            val_loss=validation_losses,
            train_loss=train_losses,
            model_save_folder=model_save_folder, 
            verbose=verbose)
    else:
        return

    stats = trainer.fit(model, train_dataloader, val_dataloader, start_epoch)
    print(f"\n[Training completed]")
    print(f'{"Number of epochs": <21}: {epochs}')
    print(f'{"Elapsed time": <21}: {round(stats["time"], 2)}s')
    print(f'{"Best train loss": <21}: {round(stats["best_train_loss"], 5)}')
    print(f'{"Best validation loss": <21}: {round(stats["best_val_loss"], 5)}')
    print(f'{"Best epoch": <21}: {stats["best_epoch"]}')

    # evaluate
    out = trainer.pred(val_dataloader)
    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")

    # visualize predictions on validation
    if arg_plot:
        outdir = os.path.join(OUT_DIR, 'images', args['model'], run_name)
        outfile = os.path.join(outdir, dt_string+'.png')
        os.makedirs(outdir, exist_ok=True)
        plot(out[0], out[1], dataset.gt, outfile=outfile)
    
    # save model run parameters
    outdir = os.path.join(OUT_DIR, 'logs', args['model'], run_name)
    outfile = os.path.join(outdir, dt_string+'.json')
    os.makedirs(outdir, exist_ok=True)

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