from torch.utils.data import random_split, DataLoader
from datetime import datetime
import argparse
import torch
import json
import os

from src.utils import create_noisy_sinus, plot
from src.mlp.datasets import SinusDataset
from src.mlp.trainers import BPTrainer
from src.mlp.models.regression import BPSimpleRegressor, PCSimpleRegressor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, 'out')

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help=f"Model selection for experiments", choices={'reg', 'clf', 'trf'}, required=True, type=str)
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", choices={'bp', 'pc'}, required=True, type=str)
    parser.add_argument('-n','--num', help=f"Number of generared samples", required=False, default=1000, type=int)
    parser.add_argument('-l','--lr', help=f"Learning rate", required=False, default=0.001, type=float)
    parser.add_argument('-e','--epochs', help=f"Training epochs", required=False, default=300, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-v','--verbose', help=f"Verbosity level", required=False, default=0, type=int)
    parser.add_argument('-i','--init', help=f"PC initialization technique", required=False, default="forward", type=str)
    parser.add_argument('-dp','--dropout', help=f"Dropout level", required=False, default=0, type=float)
    parser.add_argument('-o','--output_dir', help=f"Output directory where training results are stored", required=False, default=None, type=str)
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
        ).to(device) 

    else:
        model = PCSimpleRegressor(
            init=init,
            dropout=dropout
        ).to(device)

    # TODO Luca mentioned adam is not suitable for PC
    # we might have to change this to SGD if it performs bad on PC
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    loss = torch.nn.MSELoss()

    print(f"[Training started]")

    if train == "bp":
        trainer = BPTrainer(optimizer=optimizer, loss=loss, device=device, epochs=epochs, verbose=verbose)
    else:
        return

    stats = trainer.fit(model, train_dataloader, val_dataloader)
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
        outdir = os.path.join(OUT_DIR, 'images', args['model'])
        outfile = os.path.join(outdir, dt_string+'.png')
        os.makedirs(outdir, exist_ok=True)
        plot(out[0], out[1], dataset.gt, outfile=outfile)
    
    # save model run
    if out_dir:
        outdir = os.path.join(OUT_DIR, 'logs', args['model'])
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