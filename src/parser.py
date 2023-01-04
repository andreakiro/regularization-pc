from easydict import EasyDict as edict
import argparse


def read_arguments():
    r"""
    Argparser used to parse user inputs and set default parameters necessary for the training.

    Returns:
        edict : Dictionary containing all parsed arguments/parameters
    """
    parser = argparse.ArgumentParser()
    
    # minimal required arguments:
    parser.add_argument('-m', '--model', help=f"Model selection for experiments", choices={'reg', 'clf'}, required=True, type=str)
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", choices={'bp', 'pc'}, required=True, type=str)
    parser.add_argument('-d', '--dataset', help='dataset to be used', choices={'sine', 'housing', 'mnist', 'fashion'}, required=True,  type=str)
    parser.add_argument('--wandb', type=str, default='offline', help='set wandb online or offline', choices={'online', 'offline'})
    parser.add_argument('--seed', help='Random seed for experiment', required=False, default=42, type=int)
    
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
    parser.add_argument('-cf','--checkpoint_frequency', help=f"checkpoint frequency in epochs", required=False, default=-1, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-ns','--nsamples', help=f"Number of generated samples for regression", required=False, default=1000, type=int)
    parser.add_argument('-lps', '--log_bs_interval', help=f"frequency of batch granularity logging", required=False, default=100, type=int)

    return edict(vars(parser.parse_args()))
    