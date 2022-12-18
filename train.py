import os, re, json, argparse, torch, random
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from easydict import EasyDict as edict
from datetime import datetime
import numpy as np
import itertools
import wandb

from src.utils import plot
from src.factory import get_factory

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR  = os.path.join(ROOT_DIR, 'out')


def read_arguments():
    parser = argparse.ArgumentParser()
    
    # minimal required arguments:
    parser.add_argument('-m', '--model', help=f"Model selection for experiments", choices={'reg', 'clf'}, required=True, type=str)
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", choices={'bp', 'pc'}, required=True, type=str)
    parser.add_argument('--seed', help='Random seed for experiment', required=False, default=42, type=int)
    parser.add_argument('-d', '--dataset', help='Dataset to use: Regression [sinus, housing], Classification [MNIST, FashionMNIST]', required=False, choices={'housing', 'sinus', 'MNIST', 'FashionMNIST'}, type=str)
    
    # common to all architectures and training modes:
    parser.add_argument('-bs','--batch-size', help=f"Batch size used for training and evaluation", required=False, default=32, type=int)
    parser.add_argument('-e','--epochs', help=f"Training epochs", required=False, default=300, type=int)
    parser.add_argument('-dp','--dropout', help=f"Dropout level", required=False, default=0, type=float)
    parser.add_argument('-l','--lr', help=f"Learning rate", required=False, default=0.001, type=float)
    parser.add_argument('-v','--verbose', help=f"Verbosity level", required=False, default=0, type=int)
    parser.add_argument('-pa','--patience', help=f"Patience for the early stopping (num of epochs)", required=False, default=100, type=int)
    parser.add_argument('-md','--min_delta', help=f"Min delta improvements for early stopping", required=False, default=1e-3, type=float)

    # optimizer selection
    parser.add_argument('--optimizer', help='Weight optimizer selection', choices={'adam', 'adagrad', 'momentum', 'rmsprop'}, required=False, default='adam', type=str)
    parser.add_argument('--weight_decay', help='L2 weight decay factor', required=False, default=0.001, type=float)
    parser.add_argument('--momentum', help='Alpha for momentum optimizer', required=False, default=0.9, type=float)
    parser.add_argument('--gamma', help='Gamma for momentum torch scheduler', required=False, default=0.5, type=float)

    # PC training mode specific:
    parser.add_argument('-i','--init', help=f"PC initialization technique", choices={'zeros', 'normal', 'xavier_normal', 'forward'}, required=False, default="forward", type=str)
    parser.add_argument('--x_optimizer', help='PC optimizer selection', choices={'adam', 'adagrad', 'momentum', 'rmsprop'}, required=False, default='adam', type=str)
    parser.add_argument('-c','--clr', help=f"PC convergence learning rate", required=False, default=0.2, type=float)
    parser.add_argument('--pc_weight_decay', help='L2 weight decay factor', required=False, default=0.001, type=float)
    parser.add_argument('--pc_momentum', help='Alpha for momentum optimizer', required=False, default=0.9, type=float)
    parser.add_argument('--pc_gamma', help='PC Gamma for momentum torch scheduler', required=False, default=0.5, type=float)
    parser.add_argument('-it','--iterations', help=f"PC convergence iterations", required=False, default=100, type=int)

    # io, logging and others:
    parser.add_argument('--wandb', type=str, default='offline', help='set wandb online or offline', choices={'online', 'offline'})
    parser.add_argument('-cf','--checkpoint_frequency', help=f"checkpoint frequency in epochs", required=False, default=-1, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-ns','--nsamples', help=f"Number of generated samples for regression", required=False, default=1000, type=int)
    parser.add_argument('-lps', '--log_bs_interval', help=f"frequency of batch granularity logging", required=False, default=100, type=int)

    return edict(vars(parser.parse_args()))


def create_wandb_config(args: edict):
    # identifies an experiment
    wandb_config = dict()
    
    if args.wandb == 'offline':
        os.environ['WANDB_SILENT'] = 'true'

    wandb_config = {
        # architecture
        'model': args.model,
        'training': args.training,
        # training hyperparams
        'optimizer': args.optimizer,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        # regularization params
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
    }

    if args.training == 'reg':
        wandb_config.update({
            'es_patience': args.patience,
            'es_min_delta': args.min_delta
        })

    if args.training == 'pc':
        wandb_config.update({
            'pc_init': args.init,
            'pc_optimizer': args .x_optimizer,
            'pc_clr': args.clr,
            'pc_weight_dacay': args.pc_weight_decay,
            'pc_iters': args.iterations,
        })

    if args.optimizer == 'momentum':
        wandb_config.update({
            'momentum': args.momentum,
            'gamma': args.gamma
        })

    if args.x_optimizer == 'momentum':
        wandb_config.update({
            'pc_momentum': args.pc_momentum,
            'pc_gamma': args.pc_gamma
        })

    return wandb_config


def main():

    # fetch run args
    args = read_arguments()
    wandb_config = create_wandb_config(args)
    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.checkpoint_frequency == -1: args.checkpoint_frequency = args.epochs + 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # path to saving directories
    logs_dir   = os.path.join(OUT_DIR, 'logs', args.model, dt_string)
    plots_dir  = os.path.join(OUT_DIR, 'plots', args.model, dt_string)
    models_dir = os.path.join(OUT_DIR, 'models', args.model, dt_string)
    
    args.update({
        'models_dir': models_dir,
        'logs_dir': logs_dir,
    })
    
    # safely create directories
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    if args.plot: os.makedirs(plots_dir, exist_ok=True)
    
    factory = get_factory(args, DATA_DIR, device)
    train_loader, val_loader, model, trainer, loss = factory.train_loader, factory.val_loader, factory.model, factory.trainer, factory.loss

    wandb_config.update({
        'device': device.type,
        'loss': loss._get_name()
    })

    wandb.init(
        project='bio-transformers',
        config = wandb_config,
        mode = args.wandb,
        resume = 'auto',
    )


    print(f"[Training is starting]")
    print(args)

    stats = trainer.fit(model, train_loader, val_loader)

    print(f"\n[Training is complete]")
    print(f'{"Number of epochs": <21}: {args.epochs}')
    print(f'{"Elapsed time": <21}: {round(stats["time"], 2)}s')
    print(f'{"Best train loss": <21}: {round(stats["best_train_loss"], 5)}')
    print(f'{"Best validation loss": <21}: {round(stats["best_val_loss"], 5)}')
    print(f'{"Best epoch": <21}: {stats["best_epoch"]}')


    # final evaluation or plotting
    if args.plot and args.dataset == 'sinus':
        X, y, gt = [], [], []
        for batch, gt in val_loader:
            X.append(batch.detach().numpy())
            y.append(model(batch).detach().numpy())
            gt.append(gt.detach().numpy())
            
        X, y, gt = np.concatenate(X).ravel(), np.concatenate(y).ravel(), np.concatenate(gt).ravel()
        # TODO 5 previous lines should be a defined function

        outfile = os.path.join(plots_dir, dt_string + '.png')
        plot(X, y, gt, outfile=outfile)


    # save model run parameters
    logfile = os.path.join(logs_dir, 'info.json')

    logs = {
        "model": args.model,
        "framework" : args.training,
        "seed": args.seed,
        "device" : str(device),
        "loss" : loss._get_name(),
        "optimizer" : args.optimizer,
        "epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "learning_rate" : args.lr,
        "weight_decay": args.weight_decay,
        "dropout" : args.dropout,
    }

    if args.model == 'reg': 
        logs.update({
            "nsamples" : args.nsamples,
            "es_patience": args.patience,
            "es_min_delta": args.min_delta
        })

    if args.training == "pc":
        logs.update({
            "pc_init": args.init,
            "pc_optimizer": args.x_optimizer,
            "pc_clr": args.clr,
            'pc_weight_dacay': args.pc_weight_decay,
            "pc_iterations" : args.iterations,
        })

    if args.optimizer == 'momentum':
        logs.update({
            'momentum': args.momentum,
            'gamma': args.gamma
        })

    if args.x_optimizer == 'momentum' and args.training == 'pc':
        logs.update({
            'pc_momentum': args.pc_momentum,
            'pc_gamma': args.pc_gamma
        })

    logs.update({ "results": stats })

    with open(logfile, 'w') as f:
        json.dump(logs, f, indent=2)


if __name__ == "__main__":
    main()