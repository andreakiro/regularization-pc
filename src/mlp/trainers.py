import torch
import torch.nn as nn
import numpy as np
import time

class BPTrainer():

    def __init__(
        self,
        optimizer: torch.optim,
        loss: torch.nn.modules.loss,
        device: torch.device = torch.device('cpu', 0),
        epochs: int = 50,
        verbose: int = 0
    ) -> None:

        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.epochs = epochs
        self.device = device

        self.train_loss = []
        self.val_loss = []

    def fit(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader
    ) -> dict:

        self.model = model.to(self.device)
        start = time.time()

        for epoch in range(self.epochs):

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
        
