from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from abc import ABC

import os
import torch

from src.utils import create_noisy_sinus
from src.optimizer import set_optimizer
from src.utils import create_noisy_sinus, create_house_price
from src.mlp.datasets import SinusDataset, HousePriceDataset
from src.mlp.trainers import BPTrainer, PCTrainer
from src.mlp.models.regression import BPSimpleRegressor, PCSimpleRegressor
from src.mlp.models.classification import BPSimpleClassifier, PCSimpleClassifier

def get_factory(args, DATA_DIR, device):
    if args.model == 'reg':
        factory = RegressionFactory(args, DATA_DIR, device)

    elif args.model == 'clf':
        factory = ClassificationFactory(args, DATA_DIR)

    elif args.model == 'trf':
        raise NotImplementedError("Transformer models are not implemented yet")
    return factory
   

class Factory(ABC):
    def __init__(self, args, DATA_DIR, device):
        self.args = args
        self.DATA_DIR = DATA_DIR
        self.loss = None
        self.device = device
        
    def get_trainer(self):
        self.optimizer = set_optimizer(
            paramslist=torch.nn.ParameterList(self.model.parameters()),
            optimizer=self.args.optimizer,
            lr=self.args.lr,
            wd=self.args.weight_decay,
            mom=self.args.momentum
        )
        # TODO Luca mentioned adam is not suitable for PC
        # we might have to change this to SGD if it performs bad on PC
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

class RegressionFactory(Factory):
    def __init__(self, args, DATA_DIR, device):
        super().__init__(args, DATA_DIR, device)
        if not args.dataset or args.dataset=="sinus": # use standard (noisy sinus)
            create_noisy_sinus(num_samples=args.nsamples)
            dpath = os.path.join(DATA_DIR, 'regression', 'noisy_sinus.npy')
            self.dataset = SinusDataset(data_path=dpath, device=device)
        
        elif args.dataset == 'housing':
            create_house_price()
            dpath = os.path.join(DATA_DIR, 'regression')
            self.dataset = HousePriceDataset(data_path = dpath, device=device)
        
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_data, val_data = random_split(self.dataset, [train_size, val_size], generator=torch.Generator())
        
        self.train_loader = DataLoader(train_data, batch_size=args.batch_size)
        self.val_loader = DataLoader(val_data, batch_size=args.batch_size)
        
        if args.training == 'bp':
            self.model = BPSimpleRegressor(dropout=args.dropout, input_dim = self.dataset.sample_size)
        elif args.training == 'pc':
            self.model = PCSimpleRegressor(dropout=args.dropout, input_dim = self.dataset.sample_size)
            self.model.to(device)
                
        self.loss = torch.nn.MSELoss()
        self.get_trainer()


class ClassificationFactory(Factory):
    def __init__(self, args, DATA_DIR, device):
        super().__init__(args, DATA_DIR, device)
        if not args.dataset or args.dataset == "MNIST":
            train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
            val_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
        
        elif args.dataset == "FashionMNIST":
            train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
            val_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
        
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
        
        if args.training == 'bp': 
            self.model = BPSimpleClassifier(dropout=args.dropout)
        elif args.training == 'pc': 
            self.model = PCSimpleClassifier(dropout=args.dropout)
        self.model.to(device)
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.get_trainer()