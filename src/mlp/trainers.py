import itertools
import numpy as np
import torch.nn as nn
import torch
import wandb
import time
import os

from easydict import EasyDict as edict
from abc import ABC, abstractclassmethod
from src.optimizer import set_optimizer
from src.utils import get_out_of_distribution_sinus, plot, augment
from src.mlp.datasets import SinusDataset


class Trainer(ABC):

    @staticmethod
    def evaluate_generalization(
        dataset: str,
        model: nn.Module,
        loss: torch.nn.modules.loss,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu', 0)
    ) -> float:
        
        model.eval()

        with torch.no_grad():

            if dataset == 'sine':

                x, gt = SinusDataset.out_of_sample(100, -3, 12)
                samples_X = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
                samples_gt = torch.tensor(gt, dtype=torch.float32).unsqueeze(1).to(device)

                losses = []
                for idx, sample_x in enumerate(samples_X):
                    ground_truth = samples_gt[idx]
                    yhat = model(sample_x.to(device))
                    l = loss(yhat, ground_truth)
                    losses.append(l.detach().cpu().numpy())

                return float(np.average(losses).item())

            
            if dataset == 'mnist' or dataset == 'fashion':  
                # this is way too slow!   
                
                losses = []
                for X_val, y_val in val_loader:

                    scores, x_val = [], []

                    for sample in X_val:
                        x_aug = augment(sample.cpu().numpy()) # .to(device) here instead?
                        x_val.append(torch.Tensor(x_aug))

                    x_vals = torch.stack(x_val)
                    score = model(x_vals.to(device))
                    scores.append(score)

                    l = loss(torch.cat(scores), y_val)
                    losses.append(l.detach().cpu().numpy())

                return float(np.average(losses).item())



class BPTrainer(Trainer):

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
        plots_dir
    ):

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.plots_dir = plots_dir
        
        early_stopper = EarlyStopper(
            patience=self.args.patience, 
            min_delta=self.args.min_delta
        )

        if self.args.optimizer == 'momentum':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, 
                milestones=[24, 36, 48, 66, 72],
                gamma=self.args.gamma
            )

        wandb.watch(self.model)
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

                if np.isnan(tmp_loss[-1]):
                    print('[Abort program] Loss was nan at some point')
                    wandb.finish() 
                    exit(1)

                if self.args.verbose and (batch_idx+1) % self.args.log_bs_interval == 0:
                    print("[Epoch %d/%d] train loss: %.5f [batch %d/%d]" 
                    % (epoch+1, self.epochs, tmp_loss[-1], batch_idx * len(y_train), len(self.train_loader.dataset)))

            self.train_loss.append(np.average(tmp_loss))

            if self.args.optimizer == 'momentum':
                self.scheduler.step()

            # in: eval loop
            self.model.eval()
            with torch.no_grad():
                tmp_loss = []
                
                for X_val, y_val in self.val_loader:
                    score = self.model(X_val)
                    loss = self.loss(input=score, target=y_val)
                    tmp_loss.append(loss.detach().cpu().numpy())

                self.val_loss.append(np.average(tmp_loss))

            # watch metrics in wandb
            wandb.log({'train_loss': self.train_loss[-1], 'epoch': epoch})
            wandb.log({'test_loss': self.val_loss[-1], 'epoch': epoch})

            # log epoch summary
            if self.args.verbose:
                print("[Epoch %d/%d] train loss: %.5f, test loss: %.5f"
                % (epoch+1, self.epochs, self.train_loss[-1], self.val_loss[-1]))
                if self.args.model != 'reg': print('')

            # save model to disk
            epoch = epoch + 1 # for simplicity
            if (epoch % self.args.checkpoint_frequency == 0) or (epoch == self.epochs):
                filename = f'epoch_{epoch}.pt' if epoch != self.epochs else 'last_model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, os.path.join(self.args.models_dir, filename))

            # optional early stopping
            if early_stopper.verify(self.val_loss[-1]):
                print(f'[Early stop] val loss did not improve of more than \
                {early_stopper.min_delta} for last {early_stopper.patience} epochs')
                wandb.finish()
                break

        
        end = time.time()

        np.save(file = os.path.join(self.args.logs_dir, "train_loss.npy"), arr = np.array(self.train_loss))
        np.save(file = os.path.join(self.args.logs_dir, "val_loss.npy"), arr = np.array(self.val_loss))

        generalization_error = self.evaluate_generalization(
            dataset=self.args.dataset,
            model=self.model,
            loss=self.loss,
            val_loader=self.val_loader, 
            device=self.device
        )
        
        stats = edict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        if generalization_error is not None:
            stats['generalization'] = generalization_error

        wandb.finish()

        return stats



class PCTrainer(Trainer):
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
        self.train_energy = []

        self.init = init
        self.clr = clr
        self.iterations = iterations
    

    def fit(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        method: str = "torch"
    ) -> edict:
        """
        Fit the model.

        Parameters
        ----------
        model : nn.Module
                model to optimize

        train_dataloader : torch.utils.data.DataLoader
                           dataloader for training data
        
        val_dataloader : torch.utils.data.DataLoader
                         dataloader for validation data
        
        method : str 
                 method used to optimize the model; possible parameters are "torch", if the optimization is carried out 
                 using standard torch optimizers, or "custom", if the optimization has to be perfomed used the custom
                 backward() and step() methods implemented for the PC models; default is "torch"

        Returns
        -------
        Returns a dictionary containing the statistics on training and evaluation of the model.

        """
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # note that this optimizer only knows about linear parameters 
        # because pc parameters have not been initialized yet
        self.w_optimizer = self.optimizer 

        if self.args.optimizer == 'momentum':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer, 
                milestones=[24, 36, 48, 66, 72],
                gamma=self.args.gamma
            )

        wandb.watch(self.model)
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

                self.x_optimizer = set_optimizer(
                    paramslist=torch.nn.ParameterList(itertools.chain(*pc_parameters)), 
                    optimizer=self.args.x_optimizer,
                    lr=self.args.clr,
                    wd=self.args.pc_weight_decay,
                    mom=self.args.pc_momentum
                )

                if self.args.x_optimizer == 'momentum':
                    self.x_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer=self.x_optimizer, 
                        milestones=[24, 36, 48, 66, 72],
                        gamma=self.args.pc_gamma
                    )
                
                # fix last pc layer to output
                self.model.fix_output(y_train)
                self.model.pc_layers[-1].x.requires_grad = False
                
                # convergence step
                for _ in range(self.iterations):

                    if method == "torch":
                        
                        self.x_optimizer.zero_grad()

                        # do a pc forward pass
                        self.model.forward(X_train)

                        energy = self.model.get_energy()
                        energy.sum().backward()
                        self.x_optimizer.step()

                        if self.args.x_optimizer == 'momentum':
                            self.x_scheduler.step()
                    
                    elif method == "custom":
                            
                        # do a pc forward pass
                        self.model.forward(X_train)
                        self.model.backward_x()
                        self.model.step_x(η=0.2)

                # weight update step
                if method == "torch":
                    self.w_optimizer.step()
                elif method == "custom":
                    self.model.backward_w()
                    self.model.step_w(η=0.2)   

                # do a regular forward pass for evaluation
                self.model.eval()
                score = self.model(X_train)

                loss = self.loss(input=score, target=y_train)
                energy = self.model.get_energy()
                tmp_loss.append(loss.detach().cpu().numpy())
                tmp_energy.append(energy.detach().cpu().numpy())

                if np.isnan(tmp_loss[-1]):
                    print('[Abort program] Loss was nan at some point')
                    wandb.finish() 
                    exit(1)

                if self.args.verbose and (batch_idx+1) % self.args.log_bs_interval == 0: 
                    print("[Epoch %d/%d] train loss: %.5f train energy: %.5f [batch %d/%d]" 
                    % (epoch+1, self.epochs, tmp_loss[-1], np.average(tmp_energy[-1]),
                    batch_idx * len(y_train), len(train_loader.dataset)))

            self.train_loss.append(np.average(tmp_loss))
            self.train_energy.append(np.average(tmp_energy))

            if self.args.optimizer == 'momentum':
                self.scheduler.step()


            # in: eval loop
            self.model.eval()
            
            with torch.no_grad():

                tmp_loss = []

                for X_val, y_val in val_loader:

                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    
                    # do a regular forward pass for evaluation
                    score = self.model(X_val)
                    loss = self.loss(input=score, target=y_val)
                    tmp_loss.append(loss.detach().cpu().numpy())

                self.val_loss.append(np.average(tmp_loss))

            # watch metrics in wandb
            wandb.log({'train_loss': self.train_loss[-1], 'epoch': epoch})
            wandb.log({'train_energy': self.train_energy[-1], 'epoch': epoch})
            wandb.log({'test_loss': self.val_loss[-1], 'epoch': epoch})
                
            # log epoch summary
            if self.args.verbose:
                print("[Epoch %d/%d] train loss: %.5f train energy: %.5f test loss: %.5f" \
                    % (epoch+1, self.epochs, self.train_loss[-1], self.train_energy[-1], self.val_loss[-1]))

            # save model to disk
            epoch = epoch + 1 # for simplicity
            if (epoch % self.args.checkpoint_frequency == 0) or (epoch == self.epochs):
                filename = f'epoch_{epoch}.pt' if epoch != self.epochs else 'last_model.pt'
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

        generalization_error = self.evaluate_generalization(
            dataset=self.args.dataset,
            model=self.model,
            loss=self.loss,
            val_loader=self.val_loader, 
            device=self.device
        )

        stats = edict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        if generalization_error is not None:
            stats['generalization'] = generalization_error

        wandb.finish()

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