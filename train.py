import os, json, torch, random
from datetime import datetime
import numpy as np
import wandb

from src.factory import TrainerFactory
from src.wnb_setup import create_wandb_config
from src.parser import read_arguments
from src.utils import plot

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR  = os.path.join(ROOT_DIR, 'out')


def main():

    # fetch run args
    args = read_arguments()
    wandb_config = create_wandb_config(args)
    dt_string = datetime.now().strftime("%Y%m%d%H%M%S")

    if args.checkpoint_frequency == -1: 
        args.checkpoint_frequency = args.epochs + 1

    # set seed for control
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # path to saving directories
    logs_dir   = os.path.join(OUT_DIR, 'logs', args.model, dt_string)
    plots_dir  = os.path.join(OUT_DIR, 'plots', args.model, dt_string)
    models_dir = os.path.join(OUT_DIR, 'models', args.model, dt_string)
    
    args.update({
        'models_dir': models_dir,
        'plots_dir': plots_dir,
        'logs_dir': logs_dir,
    })
    
    # safely create directories
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    if args.plot: os.makedirs(plots_dir, exist_ok=True)

    # define model, trainer and co.
    factory = TrainerFactory(args, DATA_DIR, device)
    train_loader = factory.train_loader
    val_loader = factory.val_loader
    trainer = factory.trainer
    model = factory.model
    loss = factory.loss

    # wandb initialization

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

    # ============= training =============

    print(f"[Training is starting]")

    stats = trainer.fit(model, train_loader, val_loader, plots_dir)

    print(f"\n[Training is complete]")
    print(f'{"Number of epochs": <21}: {args.epochs}')
    print(f'{"Elapsed time": <21}: {round(stats["time"], 2)}s')
    print(f'{"Best train loss": <21}: {round(stats["best_train_loss"], 5)}')
    print(f'{"Best validation loss": <21}: {round(stats["best_val_loss"], 5)}')
    print(f'{"Best epoch": <21}: {stats["best_epoch"]}')

    if 'generalization' in stats.keys():
        print(f'{"Generalization error": <21}: {round(stats["generalization"], 5)}')

    # =========== end training ===========

    if args.model == 'reg' and args.dataset == 'sine' and args.plot:
        X, y, gt = np.concatenate(X).ravel(), np.concatenate(y).ravel(), np.concatenate(gt).ravel()
        outfile = os.path.join(plots_dir, 'noisy_sinus_plot.png')
        os.makedirs(plots_dir, exist_ok=True)
        plot(X, y, gt, outfile=outfile)


    # save model run parameters
    logfile = os.path.join(logs_dir, 'info.json')

    logs = {
        "model": args.model,
        "dataset": args.dataset,
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
