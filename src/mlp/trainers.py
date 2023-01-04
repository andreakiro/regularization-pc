import itertools
import numpy as np
import torch.nn as nn
import torch
import wandb
import time
import os

from easydict import EasyDict as edict
from abc import ABC
from src.optimizer import set_optimizer


class Trainer(ABC):
    r"""
    Abstract parent trainer class, used to contain the generalization
    method needed in all child classes.

    """
    @staticmethod
    def evaluate_generalization(
        model: nn.Module,
        loss: torch.nn.modules.loss,
        gen_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu', 0)
    ) -> float:
        r"""
        Function for testing the performance of the model on OOD-data.

        Parameters:
        ----------
            model : nn.Module
                    torch Model on which to test the ood-data on.
            loss : torch.nn.modules.loss
                    error function on the models prediction and groundtruth.
            gen_loader : torch.utils.data.DataLoader: 
                    Dataloader containing the OOD data.
            device : torch.device (optional)
                    Torch device to test the model on. Defaults to torch.device('cpu', 0).

        Returns:
        -------
            float: Averaged loss over all samples in the gen_loader.
        """
        model.eval()

        with torch.no_grad():

            if gen_loader is not None:
                losses = []
                for sample_x, ground_truth in gen_loader:
                    yhat = model(sample_x.to(device))
                    l = loss(yhat, ground_truth)
                    losses.append(l.detach().cpu().numpy())

                return float(np.average(losses).item())


class BPTrainer(Trainer):
    r"""
    Trainer class for performing backpropagation training as well as 
    testing and evaluation on the test and generalization datasets.

    Parameters:
    ----------
        args : edict
                Arguments given by the user input or standard values.
        epochs : int
                Number of epochs for training.
        optimizer : torch.optim
                The used optimizer for backpropagation.
        loss : torch.nn.modules.loss
                error function on the models prediction and groundtruth.
        device : torch.device (optional)
                Torch device to test the model on. Defaults to torch.device('cpu', 0).
    """

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
        self.gen_error = []
        self.l2 = []

    def fit(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader
    ):
        """
        Trains the model on the train_loader data, tests its performance on
        the val_loader after every epoch and measures the generalization error
        on the gen_loader data.

        Args:
            model : nn.Module
                    The torch model to train.
            train_loader : torch.utils.data.DataLoader
                    The Dataloader providing the data for training.
            val_loader : torch.utils.data.DataLoader
                    The Dataloader providing the data for testing. 
            gen_loader : torch.utils.data.DataLoader
                    The dataloader providing the data for the generalization error.

        Returns:
            edict: Dictionary containing traning metrics
        """
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gen_loader = gen_loader

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
                X_train, y_train = X_train.to(
                    self.device), y_train.to(self.device)

                self.optimizer.zero_grad()
                score = self.model(X_train)
                loss = self.loss(input=score, target=y_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 10)  # clip gradients
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
            w_parameters = [layer.weight.clone().detach()
                            for layer in self.model.linear_layers]
            l2_norm = 0.0
            for w in w_parameters:
                l2_norm += torch.sum(torch.square(w)).numpy()

            self.l2.append(l2_norm)

            self.model.eval()
            with torch.no_grad():
                tmp_loss = []

                for X_val, y_val in self.val_loader:
                    score = self.model(X_val)
                    loss = self.loss(input=score, target=y_val)
                    tmp_loss.append(loss.detach().cpu().numpy())

                self.val_loss.append(np.average(tmp_loss))

            self.gen_error.append(self.evaluate_generalization(
                model=self.model,
                loss=self.loss,
                gen_loader=self.gen_loader,
                device=self.device
            ))

            # watch metrics in wandb
            wandb.log({'train_loss': self.train_loss[-1], 'epoch': epoch})
            wandb.log({'test_loss': self.val_loss[-1], 'epoch': epoch})
            wandb.log({'gen_error': self.gen_error[-1], 'epoch': epoch})
            wandb.log({'l2_norm': self.l2[-1], 'epoch': epoch})

            # log epoch summary
            if self.args.verbose:
                print("[Epoch %d/%d] train loss: %.5f, test loss: %.5f"
                      % (epoch+1, self.epochs, self.train_loss[-1], self.val_loss[-1]))
                if self.args.model != 'reg':
                    print('')

            # save model to disk
            epoch = epoch + 1  # for simplicity
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
                break

        end = time.time()

        np.save(file=os.path.join(self.args.logs_dir, "train_loss.npy"),
                arr=np.array(self.train_loss))
        np.save(file=os.path.join(self.args.logs_dir, "val_loss.npy"),
                arr=np.array(self.val_loss))

        # generalization_error = self.evaluate_generalization(
        #     dataset=self.args.dataset,
        #     model=self.model,
        #     loss=self.loss,
        #     gen_loader=self.gen_loader,
        #     device=self.device
        # )

        stats = edict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats["generalization"] = float(min(self.gen_error))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        # if generalization_error is not None:
        #     stats['generalization'] = generalization_error
        #     wandb.run.summary["generalization_error"] = generalization_error

        wandb.finish()

        return stats


class PCTrainer(Trainer):
    """
    Class for training a PC network.

    Parameters
    ----------
        args : edict
                Arguments given by the user input or standard values.
        epochs : int
                Number of epochs for training.
        optimizer : torch.optim
                The used optimizer for weight backpropagation.
        loss : torch.nn.modules.loss
                error function on the models prediction and groundtruth.
        device : torch.device (optional)
                Torch device to test the model on. Defaults to torch.device('cpu', 0).
        init : str
              initialization technique PC hidden values; supported techniques:
                - 'zeros', hidden values initialized with 0s
                - 'normal', hidden values initialized with a normal distribution with μ=mean and σ=std
                - 'xavier_normal', hidden values initialize with values according to the method described in 
                  *Understanding the difficulty of training deep feedforward neural networks* - Glorot, X. & Bengio, Y. 
                  (2010), using a normal distribution. 
                - 'forward', hidden values initialized with the forward pass value
        clr : float
                PC learning rate
        iterations : int
                PC number of iterations for the inner loop of the algorithm

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
        self.gen_error = []
        self.l2 = []

        self.init = init
        self.clr = clr
        self.iterations = iterations

    def fit(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
        method: str = "torch"
    ) -> edict:
        """
        Trains the model on the train_loader data, tests its performance on
        the val_loader after every epoch and measures the generalization error
        on the gen_loader data.        

        Parameters
        ----------
            model : nn.Module
                    The torch model to train.
            train_loader : torch.utils.data.DataLoader
                    The Dataloader providing the data for training.
            val_loader : torch.utils.data.DataLoader
                    The Dataloader providing the data for testing. 
            gen_loader : torch.utils.data.DataLoader
                    The dataloader providing the data for the generalization error.              
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
        self.gen_loader = gen_loader

        # note that this optimizer only knows about linear parameters
        # because pc parameters have not been initialized yet
        self.w_optimizer = self.optimizer

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

            tmp_loss = []
            tmp_energy = []

            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                # in: train loop
                self.model.train()

                X_train, y_train = X_train.to(
                    self.device), y_train.to(self.device)
                self.model.reset_dropout_masks()

                self.w_optimizer.zero_grad()

                # do a pc forward pass to initialize pc layers
                self.model.forward(X_train, self.init)

                pc_parameters = [layer.parameters()
                                 for layer in self.model.pc_layers]

                self.x_optimizer = set_optimizer(
                    paramslist=torch.nn.ParameterList(
                        itertools.chain(*pc_parameters)),
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
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 10)  # clip gradients
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
            w_parameters = [layer.weight.clone().detach()
                            for layer in self.model.linear_layers]
            l2_norm = 0.0
            for w in w_parameters:
                l2_norm += torch.sum(torch.square(w)).numpy()

            self.l2.append(l2_norm)

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

            self.gen_error.append(self.evaluate_generalization(
                model=self.model,
                loss=self.loss,
                gen_loader=self.gen_loader,
                device=self.device
            ))

            # watch metrics in wandb
            wandb.log({'train_loss': self.train_loss[-1], 'epoch': epoch})
            wandb.log({'train_energy': self.train_energy[-1], 'epoch': epoch})
            wandb.log({'test_loss': self.val_loss[-1], 'epoch': epoch})
            wandb.log({'gen_error': self.gen_error[-1], 'epoch': epoch})
            wandb.log({'l2_norm': self.l2[-1], 'epoch': epoch})

            # log epoch summary
            if self.args.verbose:
                print("[Epoch %d/%d] train loss: %.5f train energy: %.5f test loss: %.5f"
                      % (epoch+1, self.epochs, self.train_loss[-1], self.train_energy[-1], self.val_loss[-1]))

            # save model to disk
            epoch = epoch + 1  # for simplicity
            if (epoch % self.args.checkpoint_frequency == 0) or (epoch == self.epochs):
                filename = f'epoch_{epoch}.pt' if epoch != self.epochs else 'last_model.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'w_optimizer_state_dict': self.w_optimizer.state_dict(),
                    'x_optimizer_state_dict': self.x_optimizer.state_dict()
                }, os.path.join(self.args.models_dir, filename))

            # optional early stopping
            if early_stopper.verify(self.val_loss[-1]):
                print(f'[Early stop] val loss did not improve of more than \
                {early_stopper.min_delta} for last {early_stopper.patience} epochs')
                break

        end = time.time()

        np.save(file=os.path.join(self.args.logs_dir, "train_loss.npy"),
                arr=np.array(self.train_loss))
        np.save(file=os.path.join(self.args.logs_dir, "val_loss.npy"),
                arr=np.array(self.val_loss))
        np.save(file=os.path.join(self.args.logs_dir,
                "train_energy.npy"), arr=np.array(self.train_loss))
        np.save(file=os.path.join(self.args.logs_dir,
                "val_energy.npy"), arr=np.array(self.val_loss))

        # generalization_error = self.evaluate_generalization(
        #     dataset=self.args.dataset,
        #     model=self.model,
        #     loss=self.loss,
        #     gen_loader=self.gen_loader,
        #     device=self.device
        # )

        stats = edict()
        stats["best_val_loss"] = float(min(self.val_loss))
        stats["best_train_loss"] = float(min(self.train_loss))
        stats['generalization'] = float(min(self.gen_error))
        stats["best_epoch"] = int(np.argmin(self.val_loss))+1
        stats['time'] = end - start

        # if generalization_error is not None:
        #     stats['generalization'] = generalization_error
        #     wandb.run.summary["generalization_error"] = generalization_error

        wandb.finish()

        return stats


class EarlyStopper:
    r"""
    Early Stopper class to calculate the change in test_loss over past epochs.

    Parameters:
    ----------
        patience : int (optional)
            The number of epochs the model has time to improve. Defaults to 10.
        min_delta : int (optional)
            The factor the test loss has to improve. Defaults to 0.

    """
    # credits: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def verify(self, validation_loss):
        r"""
        Calculates early stopping, whether the model has improved enough.

        Parameters:
        ----------
            validation_loss : float
                    The validation loss of the model in the current epoch. 

        Returns:
        -------
            boolean: True if early stopping is induced.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
