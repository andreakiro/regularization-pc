# file for Transformer and used subclasses
import math
import torch
from torch import nn, Tensor

class BPTransformer(torch.nn.Module):
    """
    Monolingual Transformer using Backpropagation
    Produce a probability distribution over output words via a log-softmax function over the decoder output
    """

    def __init__(self, d_model:int, max_input_len:int, num_heads:int, enc_layers:int, dim_ffnn:int, cls_pos:int = 0):
        """
        Args:
            d_model: len of vectorized word (embedding size)
            max_input_len: Maximum amount of words of the input sentence
            num_heads: number of heads in the multi head attention modules
            enc_layers: number of sub-layers of the encoder
            dim_ffnn: dim of the feedforward network model in the encoder layers
            cls_pos: output position to be used for decoder
            
            TODO: below params not implemented yet --> do we need this?
            scaled: boolean flag to specify whether to use normal or scaled encoder layer.
            eps: the eps value in layer normalization components.
            
        """
        super().__init__()
        torch.manual_seed(0) # for reproducability
        self.cls_pos = cls_pos # TODO: currently decoder takes all positions (cls_pos not used atm)
        self.max_input_len = max_input_len
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(size_pe=d_model, max_len=max_input_len)
        
        # encoder
        encoder_layer = EncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_ffnn, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers, enable_nested_tensor=False)

        # final linear decoder layer for the output, only predicts 
        self.decoder = nn.Linear(d_model*max_input_len, d_model)

    def forward(self, w:Tensor, padding_mask, src_mask=None):
        """
        Args:
            w: vectorized word
            src_mask: aka attention mask - mask all words after seed that have not been predicted by the transformer yet (shape: (max_input_len x max_input_len))
                      only needed during training
            padding_mask: paddings mask
        Returns:
            single output from the output layer at specified position.
        """
        # concatenate word embeddings and positional embeddings
        """print("padding_mask: ", padding_mask)
        print("src_mask: ", src_mask)
        print("input: ", w)"""
        x = w + self.pos_encoder(w)
        #print("pos_encod: ", x)
        # encoder transformation
        #print("Is training? ", self.training)
        if self.training:
            y = self.encoder(x, mask=src_mask, src_key_padding_mask=padding_mask)
            # print("y: ", y)
        else:
            y = self.encoder(x, src_key_padding_mask=padding_mask)
        z = self.decoder(y.flatten(start_dim=1)) # flatten everything after batch_dim
        # print("ouput: ", z)
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
        pe[:, 0, 0::2] += torch.cos(pos * div_term)

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