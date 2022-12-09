from easydict import EasyDict as edict
import itertools
import numpy as np
import torch.nn as nn
import torch
import time
import os


class BPTrainer():

    def __init__(
        self,
        args: edict,
        epochs: int,
        optimizer: torch.optim,
        loss: torch.nn.modules.loss,
        device: torch.device = torch.device('cpu', 0)
    ):

        self.args = args
        self.device = device
        
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss

        self.train_loss = []
        self.val_loss = []


    def fit(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        early_stopper = EarlyStopper(
            patience=self.args.patience, 
            min_delta=self.args.min_delta
        )

        start = time.time()

        for epoch in range(self.epochs):

            # in: train loop
            self.model.train()
            tmp_loss = []

            for batch_idx, (X_train, y_train) in enumerate(self.train_loader):
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)

                self.optimizer.zero_grad()
                score = self.model(X_train)
                loss = self.loss(input=score, target=y_train)
                loss.backward()
                self.optimizer.step()

                tmp_loss.append(loss.detach().cpu().numpy())

                if self.args.verbose and (batch_idx+1) % self.args.log_bs_interval == 0:
                    print("[Epoch %d/%d] train loss: %.5f [batch %d/%d]" 
                    % (epoch+1, self.epochs, tmp_loss[-1], batch_idx * len(y_train), len(self.train_loader.dataset)))

            self.train_loss.append(np.average(tmp_loss))

            # in: eval loop
            self.model.eval()
            with torch.no_grad():
                tmp_loss = []
                
                for X_val, y_val in self.val_loader:
                    score = self.model(X_val)
                    loss = self.loss(input=score, target=y_val)
                    tmp_loss.append(loss.detach().cpu().numpy())

                self.val_loss.append(np.average(tmp_loss))

            # log epoch summary
            if self.args.verbose:
                print("[Epoch %d/%d] train loss: %.5f, test loss: %.5f"
                % (epoch+1, self.epochs, self.train_loss[-1], self.val_loss[-1]))
                if self.args.model != 'reg': print('')

            # save model to disk
            if (epoch % self.args.checkpoint_frequency == 0) or (epoch == self.epochs - 1):
                filename = f'epoch_{epoch}.pt' if epoch != self.epochs - 1 else 'model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, os.path.join(self.args.models_dir, filename))

            # optional early stopping
            if early_stopper.verify(self.val_loss[-1]):
                print(f'[Early stop] val loss did not improve of more than \
                {early_stopper.min_delta} for last {early_stopper.patience} epochs')
                break
        
        end = time.time()
        
        np.save(file = os.path.join(self.args.logs_dir, "train_loss.npy"), arr = np.array(self.train_loss))
        np.save(file = os.path.join(self.args.logs_dir, "val_loss.npy"), arr = np.array(self.val_loss))

        stats = edict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        return stats



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
        args: edict,
        epochs: int,
        optimizer: torch.optim,
        loss: torch.nn.modules.loss,
        device: torch.device = torch.device('cpu', 0),
        init: str = 'forward',
        clr: float = 0.2,
        iterations: int = 100,
    ) -> None:

        self.args = args
        self.device = device
        
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss

        self.train_loss = []
        self.val_loss = []

        self.train_loss = []
        self.train_energy = []

        self.init = init
        self.clr = clr
        self.iterations = iterations
    

    def fit(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> edict:

        self.model = model.to(self.device)
        self.w_optimizer = self.optimizer 
        # note that this optimizer only knows about linear parameters 
        # because pc parameters have not been initialized yet

        self.train_loader = train_loader
        self.val_loader = val_loader

        start = time.time()
        
        for epoch in range(self.epochs):

            tmp_loss = []
            tmp_energy = []

            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                
                # in: train loop
                self.model.train()

                X_train, y_train = X_train.to(self.device), y_train.to(self.device)

                self.w_optimizer.zero_grad()

                # do a pc forward pass to initialize pc layers  
                self.model.forward(X_train, self.init)

                pc_parameters = [layer.parameters() for layer in self.model.pc_layers]
                self.x_optimizer = torch.optim.SGD(itertools.chain(*pc_parameters), self.clr)
                
                # fix last pc layer to output
                self.model.fix_output(y_train)
                self.model.pc_layers[-1].x.requires_grad = False
                
                # convergence step
                for _ in range(self.iterations):

                    self.x_optimizer.zero_grad()

                    # do a pc forward pass
                    self.model.forward(X_train)

                    energy = self.model.get_energy()
                    energy.sum().backward()
                    self.x_optimizer.step()

                # weight update step
                self.w_optimizer.step()

                # do a regular forward pass for evaluation
                self.model.eval()
                score = self.model(X_train)

                loss = self.loss(input=score, target=y_train)
                energy = self.model.get_energy()
                tmp_loss.append(loss.detach().cpu().numpy())
                tmp_energy.append(energy.detach().cpu().numpy())

                if self.args.verbose and (batch_idx+1) % self.args.log_bs_interval == 0: 
                    print("[Epoch %d/%d] train loss: %.5f train energy: %.5f [batch %d/%d]" 
                    % (epoch+1, self.epochs, tmp_loss[-1], np.average(tmp_energy[-1]),
                    batch_idx * len(y_train), len(train_loader.dataset)))

            self.train_loss.append(np.average(tmp_loss))
            self.train_energy.append(np.average(tmp_energy))

            # in: eval loop
            self.model.eval()
            
            with torch.no_grad():

                tmp_loss = []
                tmp_energy = []

                for X_val, y_val in val_loader:

                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    
                    # do a regular forward pass for evaluation
                    score = self.model(X_val)
                    
                    loss = self.loss(input=score, target=y_val)
                    tmp_loss.append(loss.detach().cpu().numpy())   

                self.val_loss.append(np.average(tmp_loss))

            # log epoch summary
            if self.args.verbose:
                print("[Epoch %d/%d] train loss: %.5f train energy: %.5f test loss: %.5f" \
                    % (epoch+1, self.epochs, self.train_loss[-1], self.train_energy[-1], self.val_loss[-1]))

            # save model to disk
            if (epoch % self.args.checkpoint_frequency == 0) or (epoch == self.epochs - 1):
                filename = f'epoch_{epoch}.pt' if epoch != self.epochs - 1 else 'model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'w_optimizer_state_dict': self.w_optimizer.state_dict(),
                    'x_optimizer_state_dict': self.x_optimizer.state_dict()
                }, os.path.join(self.args.models_dir, filename))

        end = time.time()

        np.save(file = os.path.join(self.args.logs_dir, "train_loss.npy"), arr = np.array(self.train_loss))
        np.save(file = os.path.join(self.args.logs_dir, "val_loss.npy"), arr = np.array(self.val_loss))
        np.save(file = os.path.join(self.args.logs_dir, "train_energy.npy"), arr = np.array(self.train_loss))
        np.save(file = os.path.join(self.args.logs_dir, "val_energy.npy"), arr = np.array(self.val_loss))

        stats = edict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        return stats


class EarlyStopper:

    # credits: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def verify(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False