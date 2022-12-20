from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from easydict import EasyDict as edict
from abc import ABC
import torch
import os

from src.mlp.trainers import BPTrainer, PCTrainer
from src.mlp.datasets import SinusDataset, HousePriceDataset
from src.mlp.models.regression import BPSimpleRegressor, PCSimpleRegressor
from src.mlp.models.classification import BPSimpleClassifier, PCSimpleClassifier
from src.optimizer import set_optimizer



class TrainerFactory:
    
    def __init__(
        self,
        args : edict,
        data_dir: str,
        device: torch.device
    ):

        if args.model == 'reg':
            factory = RegressionFactory(args, data_dir, device)
        
        elif args.model == 'clf':
            factory = ClassificationFactory(args, data_dir, device)

        self.model = factory.model
        self.loss = factory.loss
        self.trainer = factory.trainer

        self.train_loader = factory.train_loader
        self.val_loader = factory.val_loader



class Factory(ABC):

    def __init__(
        self,
        args : edict,
        data_dir: str,
        device: torch.device
    ):

        self.args = args
        self.data_dir = data_dir
        self.device = device
        self.loss = None
        

    def _set_trainer(self):

        self.optimizer = set_optimizer(
            paramslist=torch.nn.ParameterList(self.model.parameters()),
            optimizer=self.args.optimizer,
            lr=self.args.lr,
            wd=self.args.weight_decay,
            mom=self.args.momentum
        )

        if self.args.training == 'bp':
            self.trainer = BPTrainer(
                args      = self.args,
                epochs    = self.args.epochs,
                optimizer = self.optimizer,
                loss      = self.loss,
                device    = self.device
            )

        elif self.args.training == 'pc':
            self.trainer = PCTrainer(
                args      = self.args,
                epochs    = self.args.epochs,
                optimizer = self.optimizer,
                loss      = self.loss,
                device    = self.device,
                init       = self.args.init,
                iterations = self.args.iterations,
                clr        = self.args.clr,
            )

        return self


class RegressionFactory(Factory):

    def __init__(
        self,
        args : edict,
        data_dir: str,
        device: torch.device
    ):

        super().__init__(args, data_dir, device)

        if args.dataset == 'sine':
            dpath = os.path.join(data_dir, 'regression', 'sine')
            dpath = SinusDataset.generate(dpath, args.nsamples, 0, 4)
            self.dataset = SinusDataset(data_dir=dpath, device=device)
        
        elif args.dataset == 'housing':
            dpath = os.path.join(data_dir, 'regression', 'housing')
            dpath = HousePriceDataset.generate(dpath)
            self.dataset = HousePriceDataset(data_dir=dpath, device=device)
        
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_data, val_data = random_split(self.dataset, [train_size, val_size], generator=torch.Generator())
        
        self.train_loader = DataLoader(train_data, batch_size=args.batch_size)
        self.val_loader = DataLoader(val_data, batch_size=args.batch_size)
        
        if args.training == 'bp':
            self.model = BPSimpleRegressor(
                dropout=args.dropout,
                input_dim=self.dataset.sample_size
            )

        if args.training == 'pc':
            self.model = PCSimpleRegressor(
                dropout=args.dropout,
                input_dim=self.dataset.sample_size
            )
                
        self.model.to(device)
        self.loss = torch.nn.MSELoss()
        self._set_trainer()


class ClassificationFactory(Factory):

    def __init__(
        self,
        args : edict,
        data_dir: str,
        device: torch.device
    ):

        super().__init__(args, data_dir, device)

        if args.dataset == 'mnist':
            dpath = os.path.join(data_dir, 'classification')
            self.train_dataset = datasets.MNIST(dpath, train=True, download=True, transform=transforms.ToTensor())
            self.val_dataset = datasets.MNIST(dpath, train=False, download=True, transform=transforms.ToTensor())
        
        elif args.dataset == 'fashion':
            dpath = os.path.join(data_dir, 'classification')
            self.train_dataset = datasets.FashionMNIST(dpath, train=True, download=True, transform=transforms.ToTensor())
            self.val_dataset = datasets.FashionMNIST(dpath, train=False, download=True, transform=transforms.ToTensor())
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=args.batch_size, shuffle=True)
        
        if args.training == 'bp': 
            self.model = BPSimpleClassifier(dropout=args.dropout)

        if args.training == 'pc': 
            self.model = PCSimpleClassifier(dropout=args.dropout)
        
        self.model.to(device)
        self.loss = torch.nn.CrossEntropyLoss()
        self._set_trainer()
        