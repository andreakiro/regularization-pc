from torch.utils.data import random_split, DataLoader
from datetime import datetime
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
import json
import os

from src.utils import create_noisy_sinus, plot
from src.mlp.datasets import SinusDataset

from src.mlp.trainers.regression import BPRegressionTrainer
from src.mlp.trainers.classification import BPClassificationTrainer

from src.mlp.models.regression import BPSimpleRegressor, PCSimpleRegressor
from src.mlp.models.classification import BPClassifier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, 'out')

def read_arguments():
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('-m', '--model', help=f"Model selection for experiments", choices={'reg', 'clf', 'trf'}, required=True, type=str)
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", choices={'bp', 'pc'}, required=True, type=str)
    parser.add_argument('-e','--epochs', help=f"Training epochs", required=False, default=300, type=int)
    parser.add_argument('-l','--lr', help=f"Learning rate", required=False, default=0.001, type=float)
    parser.add_argument('-s', '--save', help=f"Save logs and model archives to disk", required=False, default=True, type=bool)
    parser.add_argument('-dp','--dropout', help=f"Dropout level", required=False, default=0, type=float)
    # reg model specific (for now)
    parser.add_argument('-n','--num', help=f"Number of generared samples", required=False, default=1000, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-v','--verbose', help=f"Verbosity level", required=False, default=0, type=int)
    parser.add_argument('-i','--init', help=f"PC initialization technique", required=False, default="forward", type=str)
    # clf model specific (for now)
    parser.add_argument('-bs', '--batch_size', help=f"Training batch size", required=False, default=64)
    parser.add_argument('-g', '--gamma', help=f"Learning rate step gamma", required=False, default=0.7)
    return vars(parser.parse_args())


def main():

    ############################
    ###### Get parameters ######
    ############################

    args = read_arguments()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(42)

    lr = args['lr']
    epochs = args['epochs']
    verbose = args['verbose']
    train = args['training']
    arg_plot = args['plot']
    dropout = args['dropout']
    init = args['init']

    ############################
    ##### Set training vars ####
    ############################

    if args['model'] == 'reg':
        create_noisy_sinus(num_samples=args['num']) # create the data, if they don't exist
        data_path = os.path.join(ROOT_DIR, "src/data/noisy_sinus.npy")
        dataset = SinusDataset(data_path=data_path, device=device)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        training_data, val_data = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(training_data, batch_size=32)
        val_loader = DataLoader(val_data, batch_size=32)

        if train == 'bp':  model = BPSimpleRegressor(dropout=dropout).to(device) 
        else: model =  PCSimpleRegressor(init=init, dropout=dropout).to(device)

    elif args['model'] == 'clf':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        training_data = datasets.MNIST('local', train=True, download=True, transform=transform)
        validation_data = datasets.MNIST('local', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(training_data, **{'batch_size': args['batch_size']})
        val_loader = torch.utils.data.DataLoader(validation_data, **{'batch_size': args['batch_size']})

        if train == 'bp':  model = BPClassifier().to(device) 
        else: raise NotImplementedError

    elif args['model'] == 'trf':
        raise NotImplementedError
    
    if args['model'] == 'reg':
        # TODO Luca mentioned adam is not suitable for PC
        # we might have to change this to SGD if it performs bad on PC
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        loss = torch.nn.MSELoss()

    elif args['model'] == 'clf':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])
        loss = F.nll_loss

    ############################
    #### Start training loop ###
    ############################

    print(f"[Training started]")

    if train == "bp":
        
        if args['model'] == 'reg':
            trainer = BPRegressionTrainer(
                optimizer=optimizer, 
                loss=loss, 
                device=device, 
                epochs=epochs, 
                verbose=verbose,
            )

        elif args['model'] == 'clf':
            trainer = BPClassificationTrainer(
                optimizer=optimizer, 
                scheduler=scheduler,
                loss=loss, 
                device=device, 
                epochs=epochs, 
                verbose=verbose,
            )
        
    elif train == 'pc':
        raise NotImplementedError

    stats = trainer.fit(model, train_loader, val_loader)

    print(f"\n[Training completed]")
    print(f'{"Number of epochs": <21}: {epochs}')
    print(f'{"Elapsed time": <21}: {round(stats["time"], 2)}s')
    print(f'{"Best train loss": <21}: {round(stats["best_train_loss"], 5)}')
    print(f'{"Best validation loss": <21}: {round(stats["best_val_loss"], 5)}')
    print(f'{"Best epoch": <21}: {stats["best_epoch"]}')

    ############################
    #### Post training stuff ###
    ############################

    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")

    if arg_plot and args['model'] == 'reg':
        # visualize predictions on validation
        out = trainer.pred(val_loader)
        outdir = os.path.join(OUT_DIR, 'images', args['model'])
        outfile = os.path.join(outdir, dt_string+'.png')
        os.makedirs(outdir, exist_ok=True)
        plot(out[0], out[1], dataset.gt, outfile=outfile)
    
    if args['save']:
        # save model run logs to disk
        outdir = os.path.join(OUT_DIR, 'logs', args['model'])
        outfile = os.path.join(outdir, dt_string+'.json')
        os.makedirs(outdir, exist_ok=True)

        log = {
            "framework" : train,
            "epochs" : epochs,
            "optimizer" : type (optimizer).__name__,
            "loss" : loss._get_name() if args['model'] == 'reg' else 'todo',
            "lr" : lr,
            "results" : stats
        }

        with open(outfile, 'w') as f:
            json.dump(log, f, indent=2)

        # save model archive to disk
        outdir = os.path.join(OUT_DIR, 'models', args['model'])
        outfile = os.path.join(outdir, dt_string+'.pt')
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.state_dict(), outfile)

if __name__ == '__main__':
    main()