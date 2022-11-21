import argparse
import os
import torch
from torch.utils.data import random_split, DataLoader
import json
from datetime import datetime


from utils import create_noisy_sinus, plot, ROOT_DIR
from src.datasets import SinusDataset
from src.trainers import BPTrainer
from src.models import BPSimpleRegressor, PCSimpleRegressor


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--training', help=f"Training framework, either 'bp' (backprop) or 'pc' (predictive coding)", required=True, type=str)
    parser.add_argument('-n','--num', help=f"Number of generared samples", required=False, default=1000, type=int)
    parser.add_argument('-l','--lr', help=f"Learning rate", required=False, default=0.001, type=float)
    parser.add_argument('-e','--epochs', help=f"Training epochs", required=False, default=300, type=int)
    parser.add_argument('-p','--plot', help=f"Plot the results after training or not", required=False, default=False, type=bool)
    parser.add_argument('-v','--verbose', help=f"Verbosity level", required=False, default=0, type=int)
    parser.add_argument('-o','--output_dir', help=f"Output directory where training results are stored", required=False, default=None, type=str)
    args = vars(parser.parse_args())
    assert args['training'] in ['bp', 'pc']
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

    data_path = os.path.join(ROOT_DIR, "data/noisy_sinus.npy")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = SinusDataset(data_path=data_path, device=device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    training_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(training_data, batch_size=32)
    val_dataloader = DataLoader(val_data, batch_size=32)

    model = BPSimpleRegressor().to(device) if train == "bp" else PCSimpleRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # TODO Luca mentioned adam is not suitable for PC, we might have to change this to SGD if it performs bad on PC
    loss = torch.nn.MSELoss()

    print(f"[Training started]")
    if train == "bp":
        trainer = BPTrainer(optimizer=optimizer, loss=loss, epochs=epochs, verbose=verbose)
    else:
        pass

    stats = trainer.fit(model, train_dataloader, val_dataloader)
    print(f"\n[Training completed]")
    print(f"Number of epochs: {epochs}")
    print(f"Elapsed time: {stats['time']} s")
    print(f"Best train loss: {stats['best_val_loss']}")
    print(f"Best validation loss: {stats['best_val_loss']}")
    print(f"Best epoch: {stats['best_epoch']}")

    # evaluate
    out = trainer.pred(val_dataloader)

    # visualize predictions on validation
    if arg_plot:
        plot(out[0], out[1], dataset.gt)
    # save model run
    if out_dir:
        os.makedirs(os.path.join(ROOT_DIR, out_dir), exist_ok=True)
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        log = {
            "framework" : train,
            "epochs" : epochs,
            "optimizer" : type (optimizer).__name__,
            "loss" : loss._get_name(),
            "lr" : lr,
            "results" : stats
        }
        with open(os.path.join(os.path.join(ROOT_DIR, out_dir), dt_string+".json"), 'w') as f:
            json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()