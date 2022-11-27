# file for Transformer and used subclasses
import math
import torch
from torch import nn, Tensor

class BPTransformer(torch.nn.Module):
    """
    Monolingual Transformer using Backpropagation
    Produce a probability distribution over output words via a log-softmax function over the decoder output
    """

    def __init__(self, alphabet_size:int, d_model:int, max_input_len:int, num_heads:int, enc_layers:int, dim_ffnn:int, cls_pos:int = 0):
        """
        Args:
            alphabet_size: number of words
            d_model: dim/number of expected features in the encoder/decoder inputs
            max_input_len: Maximum length of the input sentence
            num_heads: number of heads in the multi head attention modules
            enc_layers: number of sub-layers of the encoder
            dim_ffnn: dim of the feedforward network model
            cls_pos: output position to be used to classify
            
            TODO: below params not implemented yet --> do we need this?
            scaled: boolean flag to specify whether to use normal or scaled encoder layer.
            eps: the eps value in layer normalization components.
            
        """
        super().__init__()
        torch.manual_seed(0) # for reproducability
        self.cls_pos = cls_pos
        
        # embedding and positional encoding
        self.word_embedding = torch.nn.Embedding(num_embeddings=alphabet_size, embedding_dim=d_model)
        self.pos_encoder = PositionalEncoding(size_pe=d_model, max_len=max_input_len)
        
        # encoder
        encoder_layer = EncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_ffnn)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # final linear decoder layer for the output 
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, w:Tensor):
        """
        Args:
            w: word
        Returns:
            single output from the output layer at specified position.
        """
        # concatenate word embeddings and positional embeddings
        x = self.word_embedding(w) + self.pos_encoding(len(w))
        # encoder transformation
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.decoder(y[self.cls_pos])
        return z

class PositionalEncoding(nn.Module):
    """
    positional encoding using sinus and cosinus
    inspired by https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, size_pe:int, max_len:int = 5000):
        """
        Args:
            size_pe: size of the positional encodings
            max_len: maximal length of the input sentence
        """
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, size_pe, 2) * (-math.log(10000.0) / size_pe))
        pe = torch.zeros(max_len, 1, size_pe)
        
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: Tensor of shape (sequence_len, batch_size, embedding_dim)
        """
        return x + self.pe[:x.size(0)]


class EncoderLayer(nn.TransformerEncoderLayer):
    """
    extended Encoder layer for the standard pytorch encoder module of the transformer
    """
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer.
        Args:
            src: sequence to encoder layer
            src_mask: mask for src sequence
            src_key_padding_mask: mask for the src keys for each batch
        """
        
        x = src
        # Transformer's [self-attention + add & norm]
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        # Transformer's [feedforward + add & norm]  
        x = self.norm2(x + self._ff_block(x))
        
        return x