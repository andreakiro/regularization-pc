import torch
import torch.nn as nn
import numpy as np
import time
import os

class BPTransformerTrainer():

    def __init__(
        self,
        optimizer: torch.optim,
        loss: torch.nn.modules.loss,
        model_save_folder: str,
        log_save_folder: str,
        sequence_len: int,
        checkpoint_frequency: int = 1,
        epochs: int = 50,
        early_stopping: int = 50,
        device: torch.device = torch.device('cpu', 0),
        verbose: int = 0
    ):
        """Trainer for the standard backpropagation-trained monolingual transformer

        Args:
            optimizer (torch.optim): Optimizer
            loss (torch.nn.modules.loss): Loss function
            model_save_folder (str): data path to the folder where the model and optimizer state dicts are saved (given via "run" cmd argument)
            log_save_folder (str): data path to folder where the training/validation losses as well as the training stats are saved (given via "run" cmd argument)
            sequence_len (int): the length of the data
            checkpoint_frequency (int, optional): Frequency in epochs in which the best model and optimizer parameters are saved Defaults to 1.
            epochs (int, optional): Number of training epochs. Defaults to 50.
            early_stopping (int, optional): How many past epochs are considered when comparing if the loss got worse. Defaults to 50.
            device (torch.device, optional): Device for training. Defaults to torch.device('cpu', 0).
            verbose (int, optional): Whether to print to the console or not. Defaults to 0.
        """

        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.device = device
        self.model_save_folder = model_save_folder
        self.log_save_folder = log_save_folder
        self.sequence_len = sequence_len
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping = early_stopping
        self.verbose = verbose

    def fit(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        start_epoch: int = 0,
        val_loss=[],
        train_loss=[]
    ):
        self.model = model.to(self.device)
        start = time.time()
        input_len = model.max_input_len
        source_mask = self.generate_square_subsequent_mask(input_len)

        for epoch in range(self.epochs)[start_epoch:]:
            model.train()
            tmp_loss = []
            for idx, (X_train, X_padding_mask) in enumerate(train_dataloader): # X_train is of shape (batch_size, sequence_len, embedding_size), in other words (batch_size, num_words, vectorized_word_size)
                for start_idx in range(0, self.sequence_len-input_len-1): # iterate through the sequence until there is no word left
                    sub_sequence = X_train[:, start_idx:(start_idx + input_len)].to(self.device)
                    target = X_train[:, start_idx+input_len].to(self.device) # target is next word after sub-sequence
                    self.optimizer.zero_grad()
                    prediction = model(w=sub_sequence, src_mask=source_mask, padding_mask=X_padding_mask[:, start_idx:(start_idx + input_len)])
                    loss = self.loss(input=prediction, target=target)
                    loss.backward()
                    self.optimizer.step()
                    tmp_loss.append(loss.detach().cpu().numpy())
                
                if idx >10:
                    break
                        
            train_loss.append(np.average(tmp_loss))

            model.eval()
            with torch.no_grad():
                tmp_loss = []
                for idx, (X_val, X_padding_mask) in enumerate(val_dataloader):
                    prediction = model(X_val[:, 0:input_len], padding_mask = X_padding_mask[:, 0:input_len])
                    loss = self.loss(input=prediction, target=X_val[:, input_len])
                    tmp_loss.append(loss.detach().cpu().numpy())
                    if idx > 10:
                        break
                val_loss.append(np.average(tmp_loss))
                
            if self.verbose:
                print("[Epoch %d/%d] train loss: %.5f, test loss: %.5f" % (epoch+1, self.epochs, train_loss[-1], val_loss[-1]))
            
            # checkpoint model every 30 epochs
            if (epoch % self.checkpoint_frequency == 0) or (epoch == self.epochs-1):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, os.path.join(self.model_save_folder, f"checkpoint_{epoch}.pt"))
            
            # save validation losses every epoch
            np.save(file = os.path.join(self.log_save_folder, "validation_losses.npy"), arr = np.array(val_loss))
            np.save(file = os.path.join(self.log_save_folder, "train_losses.npy"), arr = np.array(train_loss))
            
            # check for early stopping
            if self.check_early_stopping(val_loss):
                print(f"Early stopping induced, evaluation loss has not improved for the last {self.early_stopping} epochs.")
                break
        end = time.time()

        stats = dict()
        stats["best_val_loss"] = float(min(val_loss))
        stats["best_train_loss"] = float(min(train_loss))
        stats["best_epoch"] = int(np.argmin(val_loss))+1
        stats['time'] = end - start

        return stats
    
    def check_early_stopping(self, val_loss):
        if len(val_loss) <= self.early_stopping:
            return False
        else:
            return max(val_loss[-self.early_stopping-1:-2]) <= val_loss[-1]
    
    def evaluate(self, val_dataloader, model):
        val_loss = []
        model.eval()
        input_len = model.max_input_len
        with torch.no_grad():
            tmp_loss = []
            for X_val, X_padding_mask in val_dataloader:
                prediction = model(X_val[:, 0:input_len], padding_mask = X_padding_mask[:, 0:input_len])
                loss = self.loss(input=prediction, target=X_val[:, input_len])
                tmp_loss.append(loss.detach().cpu().numpy())
            val_loss.append(np.average(tmp_loss))
        return val_loss
    
    @staticmethod
    def generate_square_subsequent_mask(max_input_len):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(max_input_len, max_input_len) * float('-inf'), diagonal=0)