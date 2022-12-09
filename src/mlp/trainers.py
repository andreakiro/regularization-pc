import itertools
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
        model_dir: str,
        log_dir: str,
        cp_freq: int = 1,
        epochs: int = 50,
        early_stp: int = 50,
        device: torch.device = torch.device('cpu', 0),
        verbose: int = 0,
        val_loss: list[float] = [],
        train_loss: list[float] = []
    ):

        self.optimizer = optimizer
        self.loss = loss
        self.verbose = verbose
        self.epochs = epochs
        self.device = device
        self.model_save_folder = model_dir
        self.log_save_folder = log_dir
        self.checkpoint_frequency = cp_freq
        self.early_stopping = early_stp
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.log_bs_interval = 100


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

            self.model.train()
            tmp_loss = []
            for batch_idx, (X_train, y_train) in enumerate(train_dataloader):
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                self.optimizer.zero_grad()
                score = self.model(X_train)
                loss = self.loss(input=score, target=y_train)
                loss.backward()
                self.optimizer.step()
                tmp_loss.append(loss.detach().cpu().numpy())
                if self.verbose and (batch_idx+1) % self.log_bs_interval == 0: # self.args.log_bs_interval
                    print("[Epoch %d/%d] train loss: %.5f [Batch %d/%d]" 
                    % (epoch+1, self.epochs, tmp_loss[-1], batch_idx * len(y_train), 
                    len(train_dataloader.dataset)))
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
            
            # checkpoint model every 30 epochs
            if (epoch % self.checkpoint_frequency == 0) or (epoch == self.epochs-1):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.model_save_folder, f"checkpoint_{epoch}.pt"))
            
            # save validation losses every epoch
            np.save(file = os.path.join(self.log_save_folder, "validation_losses.npy"), arr = np.array(self.val_loss))
            np.save(file = os.path.join(self.log_save_folder, "train_losses.npy"), arr = np.array(self.train_loss))
            
            # check for early stopping
            if self.check_early_stopping():
                print(f"Early stopping induced, evaluation loss has not improved for the last {self.early_stopping} epochs.")
                break
        end = time.time()

        stats = dict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        return stats
    
    def check_early_stopping(self):
        if len(self.val_loss) <= self.early_stopping:
            return False
        else:
            return max(self.val_loss[-self.early_stopping-1:-2]) <= self.val_loss[-1]


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
        device: torch.device = torch.device('cpu'),
        init: str = 'forward',
        epochs: int = 10,
        iterations: int = 100,
        clr: float = 0.2,
        verbose: int = 0
    ) -> None:

        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.init = init
        self.epochs = epochs
        self.iterations = iterations
        self.clr = clr
        self.verbose = verbose
        self.train_loss = []
        self.train_energy = []
        self.val_loss = []
        self.val_energy = []
        self.log_bs_interval = 100
    
    def fit(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        start_epoch = 0
    ) -> dict:

        self.model = model.to(self.device)
        self.w_optimizer = self.optimizer # note that this optimizer only knows about linear parameters because pc parameters have not been initialized yet.
        
        start = time.time()
        
        for epoch in range(self.epochs):

            tmp_loss = []
            tmp_energy = []

            for batch_idx, (X_train, y_train) in enumerate(train_dataloader):
                
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

                if self.verbose and (batch_idx+1) % self.log_bs_interval == 0: # self.args.log_bs_interval
                    print("[Epoch %d/%d] train loss: %.5f train energy: %.5f [Batch %d/%d]" 
                    % (epoch+1, self.epochs, tmp_loss[-1], np.average(tmp_energy[-1]),
                    batch_idx * len(y_train), len(train_dataloader.dataset)))

            self.train_loss.append(np.average(tmp_loss))
            self.train_energy.append(np.average(tmp_energy))

            self.model.eval()
            
            with torch.no_grad():

                tmp_loss = []
                tmp_energy = []

                for X_val, y_val in val_dataloader:

                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    
                    # do a regular forward pass for evaluation
                    score = self.model(X_val)
                    
                    loss = self.loss(input=score, target=y_val)
                    tmp_loss.append(loss.detach().cpu().numpy())   

                self.val_loss.append(np.average(tmp_loss))

            if self.verbose:
                print("[Epoch %d/%d] train loss: %.5f train energy: %.5f test loss: %.5f" \
                    % (epoch+1, self.epochs, self.train_loss[-1], self.train_energy[-1], self.val_loss[-1]))

        end = time.time()

        stats = dict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        return stats
