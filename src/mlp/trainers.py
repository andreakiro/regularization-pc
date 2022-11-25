import itertools
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

            self.model.train()
            tmp_loss = []
            for X_train, y_train in train_dataloader:
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()
                score = self.model(X_train)
                loss = self.loss(input=score, target=y_train)
                loss.backward()
                self.optimizer.step()
                tmp_loss.append(loss.detach().cpu().numpy())
            self.train_loss.append(np.average(tmp_loss))

            self.model.eval()
            with torch.no_grad():
                tmp_loss = []
                for X_val, y_val in val_dataloader:
                    score = self.model(X_val)
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

    def pred(self, pred_dataloader):
        X, pred = [], []
        for batch, _ in pred_dataloader:
            for x in batch:
                X.append(x.detach().cpu().numpy())
                pred.append(self.model(x).detach().cpu().numpy())
        return X, pred

class PCTrainer():
    """
    Class for training a PC network.

    Parameters
    ----------
    init    : str
              initialization technique PC hidden values; supported techniques:
                - 'zeros', hidden values initialized with 0s
                - 'normal', hidden values initialized with a normal distribution with μ=mean and σ=std
                - 'xavier_normal', hidden values initialize with values according to the method described in 
                  *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
                  (2010), using a normal distribution. 
                - 'forward', hidden values initialized with the forward pass value

    """
    def __init__(
        self,
        optimizer,
        loss: torch.nn.modules.loss,
        device: torch.device = torch.device('cpu', 0),
        init: str = 'forward',
        epochs: int = 50,
        iterations: int = 10,
        verbose: int = 0
    ) -> None:

        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.init = init
        self.epochs = epochs
        self.iterations = iterations
        self.verbose = verbose

        self.train_loss = []
        self.val_loss = []
    
    def fit(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader
    ) -> dict:

        self.model = model.to(self.device)
        self.w_optimizer = self.optimizer # note that this optimizer only knows about linear parameters because pc parameters have not been initialized yet.
        
        start = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()    
            tmp_loss = []
            for X_train, y_train in train_dataloader:
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)

                self.w_optimizer.zero_grad()

                # initialize pc layers
                if self.init == 'forward':
                    # do a regular forward pass to get initialization values
                    self.model(X_train)
                    for pc_layer, predicted_activation in zip(
                        self.model.pc_layers, 
                        self.model.predicted_activations
                    ):
                        pc_layer.init(self.init, predicted_activation)
                else:
                    raise NotImplementedError(f"{self.init} currently not implemented")

                pc_parameters = [layer.parameters() for layer in self.model.pc_layers]
                self.x_optimizer = torch.optim.SGD(itertools.chain(*pc_parameters), 0.05)
                
                # fix last pc layer to output
                self.model.fix_output(y_train)
                self.model.pc_layers[-1].x.requires_grad = False
                
                # convergence step
                for _ in range(self.iterations):
                    self.x_optimizer.zero_grad()

                    # do a pc forward pass
                    self.model.pc_forward(X_train)

                    energy = self.model.get_energy()
                    energy.backward()
                    self.x_optimizer.step()

                # weight update step
                self.w_optimizer.step()

                tmp_loss.append(energy.detach().cpu().numpy())
            self.train_loss.append(np.average(tmp_loss))

            self.model.eval()
            
            with torch.no_grad():
                tmp_loss = []
                for X_val, y_val in val_dataloader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    
                    # do a regular forward pass
                    self.model(X_val)
                    
                    energy = self.model.get_energy()

                    tmp_loss.append(energy.detach().cpu().numpy())
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

    def pred(self, pred_dataloader):
        X, pred = [], []
        for batch, _ in pred_dataloader:
            for x in batch:
                X.append(x.detach().cpu().numpy())
                pred.append(self.model(x).detach().cpu().numpy())
        return X, pred        
