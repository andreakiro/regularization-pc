import os, re, json, argparse, torch, random
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from easydict import EasyDict as edict
from datetime import datetime
import numpy as np

from src.utils import create_noisy_sinus, plot
from src.mlp.datasets import SinusDataset
from src.mlp.trainers import BPTrainer, PCTrainer
from src.mlp.models.regression import BPSimpleRegressor, PCSimpleRegressor
from src.mlp.models.classification import BPSimpleClassifier, PCSimpleClassifier

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR  = os.path.join(ROOT_DIR, 'out')

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def read_arguments():
    parser = argparse.ArgumentParser()
    
    # minimal required arguments:
    parser.add_argument('-m', '--model', help=f"Model selection for experiments", choices={'reg', 'clf', 'trf'}, required=True, type=str)
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", choices={'bp', 'pc'}, required=True, type=str)
    
    # common to all architectures and training modes:
    parser.add_argument('-bs','--batch-size', help=f"Batch size used for training and evaluation", required=False, default=32, type=int)
    parser.add_argument('-e','--epochs', help=f"Training epochs", required=False, default=300, type=int)
    parser.add_argument('-dp','--dropout', help=f"Dropout level", required=False, default=0, type=float)
    parser.add_argument('-l','--lr', help=f"Learning rate", required=False, default=0.001, type=float)
    parser.add_argument('-v','--verbose', help=f"Verbosity level", required=False, default=0, type=int)
    parser.add_argument('-es','--early_stopping', help=f"the number of past epochs taken into account for early_stopping", required=False, default=300, type=int)

    # PC training mode specific:
    parser.add_argument('-i','--init', help=f"PC initialization technique", choices={'zeros', 'normal', 'xavier_normal', 'forward'}, required=False, default="forward", type=str)
    parser.add_argument('-c','--clr', help=f"PC convergence learning rate", required=False, default=0.2, type=float)
    parser.add_argument('-it','--iterations', help=f"PC convergence iterations", required=False, default=100, type=int)

    # io, logging and others:
    parser.add_argument('-lg','--log', help=f"Log info and results of the model or not", required=False, default=False, type=bool)
    parser.add_argument('-cf','--checkpoint_frequency', help=f"checkpoint frequency in epochs", required=False, default=1, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-ns','--nsamples', help=f"Number of generated samples for regression", required=False, default=1000, type=int)
    # parser.add_argument('-r', '--run', help=f"Individual run name, if reused the training is resumed", required=True, type=str)

    return edict(vars(parser.parse_args()))


def main():

    # fetch run args
    args = read_arguments()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")

    # path to saving directories
    logs_dir   = os.path.join(OUT_DIR, 'logs', args.model, dt_string)
    plots_dir  = os.path.join(OUT_DIR, 'plots', args.model, dt_string)
    models_dir = os.path.join(OUT_DIR, 'models', args.model, dt_string)
    
    # safely create directories
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


    if args.model == 'reg':
        create_noisy_sinus(outdir=DATA_DIR, num_samples=args.nsamples)
        dpath = os.path.join(DATA_DIR, 'regression', 'noisy_sinus.npy')
        sinus_dataset = SinusDataset(data_path=dpath, device=device)
        train_size = int(0.8 * len(sinus_dataset))
        val_size = len(sinus_dataset) - train_size
        train_data, val_data = random_split(sinus_dataset, [train_size, val_size], generator=torch.Generator())
        train_loader = DataLoader(train_data, batch_size=args.batch_size)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        
        if args.training == 'bp': model = BPSimpleRegressor(dropout=args.dropout)
        elif args.training == 'pc': model = PCSimpleRegressor(dropout=args.dropout)
        loss = torch.nn.MSELoss()

    elif args.model == 'clf':
        train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
        val_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
        
        if args.training == 'bp': model = BPSimpleClassifier(dropout=args.dropout)
        elif args.training == 'pc': model = PCSimpleClassifier(dropout=args.dropout)
        loss = torch.nn.CrossEntropyLoss()

    elif args.model == 'trf':
        raise NotImplementedError("Transformer models are not implemented yet")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO Luca mentioned adam is not suitable for PC
    # we might have to change this to SGD if it performs bad on PC
    # TODO: add reload from checkpoint (not necessary)
    
    if args.training == 'bp':
        trainer = BPTrainer(
            # essentials
            device    = device,
            epochs    = args.epochs,
            optimizer = optimizer,
            loss      = loss,
            early_stp = args.early_stopping,
            # io related
            verbose   = args.verbose,
            cp_freq   = args.checkpoint_frequency,
            model_dir = models_dir,
            log_dir   = logs_dir, 
        )

    elif args.training == 'pc':
        trainer = PCTrainer(
            # essentials
            device     = device,
            epochs     = args.epochs,
            optimizer  = optimizer, 
            loss       = loss,
            init       = args.init,
            iterations = args.iterations,
            clr        = args.clr,
            # io related
            verbose    = args.verbose
        )

    print(f"[Training is starting]")
    stats = trainer.fit(model, train_loader, val_loader, 0)

    print(f"\n[Training is complete]")
    print(f'{"Number of epochs": <21}: {args.epochs}')
    print(f'{"Elapsed time": <21}: {round(stats["time"], 2)}s')
    print(f'{"Best train loss": <21}: {round(stats["best_train_loss"], 5)}')
    print(f'{"Best validation loss": <21}: {round(stats["best_val_loss"], 5)}')
    print(f'{"Best epoch": <21}: {stats["best_epoch"]}')


    if args.plot and args.model == 'reg':
        os.makedirs(plots_dir, exist_ok=True)

        X, y = [], []
        for batch, _ in val_loader:
            X.append(batch.detach().numpy())
            y.append(model(batch).detach().numpy())
        X, y = np.concatenate(X).ravel(), np.concatenate(y).ravel()
        # TODO 5 previous lines should be a defined function

        outfile = os.path.join(os.path.dirname(plots_dir), dt_string + '.png')
        plot(X, y, sinus_dataset.gt, outfile=outfile if log else None)


    # save model run parameters
    logfile = os.path.join(logs_dir, 'runlogs.json')
    log = {
        "framework" : args.training,
        "epochs" : args.epochs,
        "optimizer" : type (optimizer).__name__,
        "loss" : loss._get_name(),
        "lr" : args.lr,
        "results" : stats
    }
    with open(logfile, 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()