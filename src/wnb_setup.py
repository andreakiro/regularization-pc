from easydict import EasyDict as edict
import os


def create_wandb_config(args: edict):
    r"""
    Creates Config File from the user input parameters, and uploads it to the 
    corresponding wandb experiment

    Args:
        args : edict
                Dictionary containing the user inputs or default values.

    Returns:
        dict: Dictionary with config
    """
    # identifies an experiment
    wandb_config = dict()

    if args.wandb == 'offline':
        os.environ['WANDB_SILENT'] = 'true'

    wandb_config = {
        # architecture
        'model': args.model,
        'dataset': args.dataset,
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
