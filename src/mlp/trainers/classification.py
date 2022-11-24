import torch
import torch.nn as nn
import numpy as np
import time

class BPClassificationTrainer():

    def __init__(
        self,
        optimizer: torch.optim,
        scheduler: torch.optim,
        loss: torch.nn.modules.loss,
        device: torch.device = torch.device('cpu', 0),
        epochs: int = 50,
        verbose: int = 0,
    ) -> None:

        self.optimizer = optimizer
        self.scheduler = scheduler
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
            for batch_idx, (X_train, y_train) in enumerate(train_dataloader):
                data, target = X_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()
                preds = model(X_train)
                loss = self.loss(input=preds, target=y_train)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(data), len(train_dataloader.dataset),
                        100. * batch_idx / len(train_dataloader), loss.item()))

            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for X_val, y_val in val_dataloader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    preds = model(data)
                    test_loss += self.loss(input=preds, target=y_train, reduction='sum').item()
                    pred = preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(val_dataloader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(val_dataloader.dataset),
                100. * correct / len(val_dataloader.dataset)))
                        
            self.scheduler.step()

        end = time.time()

        stats = dict()
        stats["best_val_loss"] = 0
        stats["best_train_loss"] = 0
        stats["best_epoch"] = 0
        stats['time'] = end - start

        return stats

    # def pred(self, pred_dataloader):
    #     X, pred = [], []
    #     for batch, _ in pred_dataloader:
    #         for x in batch:
    #             X.append(x.detach().cpu().numpy())
    #             pred.append(self.model(x).detach().cpu().numpy())
    #     return X, pred

class PCClassificationTrainer(nn.Module):

    def __init__(self):
        raise NotImplementedError()
    
    def fit(self, **kwargs):
        raise NotImplementedError()

    def pred(self, **kwargs):
        raise NotImplementedError()
        
