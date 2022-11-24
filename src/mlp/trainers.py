import torch
import torch.nn as nn
import numpy as np
import time
import os

class BPTrainer():

    def __init__(
        self,
        optimizer: torch.optim,
        loss: torch.nn.modules.loss,
        model_save_folder: str,
        log_save_folder: str,
        checkpoint_frequency: int = 1,
        device: torch.device = torch.device('cpu', 0),
        epochs: int = 50,
        verbose: int = 0,
        val_loss = [],
        train_loss = []
    ):

        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.epochs = epochs
        self.device = device
        self.model_save_folder = model_save_folder
        self.log_save_folder = log_save_folder
        self.checkpoint_frequency = checkpoint_frequency

        self.train_loss = train_loss
        self.val_loss = val_loss

    def fit(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        start_epoch: int = 0
    ):
        self.model = model.to(self.device)
        start = time.time()

        for epoch in range(self.epochs)[start_epoch:]:

            model.train()
            tmp_loss = []
            for X_train, y_train in train_dataloader:
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()
                score = model(X_train)
                loss = self.loss(input=score, target=y_train)
                loss.backward()
                self.optimizer.step()
                tmp_loss.append(loss.detach().cpu().numpy())
            self.train_loss.append(np.average(tmp_loss))

            model.eval()
            with torch.no_grad():
                tmp_loss = []
                for X_val, y_val in val_dataloader:
                    score = model(X_val)
                    loss = self.loss(input=score, target=y_val)
                    tmp_loss.append(loss.detach().cpu().numpy())
                self.val_loss.append(np.average(tmp_loss))
                        
            if self.verbose:
                print("[Epoch %d/%d] train loss: %.5f, test loss: %.5f" % (epoch+1, self.epochs, self.train_loss[-1], self.val_loss[-1]))
            
            # checkpoint model every 30 epochs
            if (epoch % self.checkpoint_frequency == 0) or (epoch == self.epochs-1):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.model_save_folder, f"checkpoint_{epoch}.pt"))
            
            # save validation losses every epoch
            np.save(file = os.path.join(self.log_save_folder, "validation_losses.npy"), arr = np.array(self.val_loss))
            np.save(file = os.path.join(self.log_save_folder, "train_losses.npy"), arr = np.array(self.train_loss))
        end = time.time()

        stats = dict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        return stats


class PCTrainer(nn.Module):

    def __init__(self):
        raise NotImplementedError()
    
    def fit(self, **kwargs):
        raise NotImplementedError()

    def pred(self, **kwargs):
        raise NotImplementedError()
        
