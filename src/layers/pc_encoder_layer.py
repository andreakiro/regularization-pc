import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoderLayer
from typing import Union, Callable, Optional
from src.layers import PCLayer
from src.layers.pc_multi_head_attention import PCMultiheadAttention


class PCTransformerEncoderLayer(TransformerEncoderLayer):
    r"""
    Custom PC Transformer Encoder Layer.

    Parameters
    ----------
    d_model : int
           The dimension of the input and the output of the encoder.
    nhead : int
            The number of PC Attention Heads.
    dim_feedforward : int (optional)
            The input dimension of the feed forward network. Defaults to 2048.
    dropout : float (optional)
            Probability of an element to be zeroed. Defaults to 0.1.
    activation : Union[str, Callable[[Tensor], Tensor]] (optional)
            Activation function of the intermediate layer. Defaults to F.relu.
    layer_norm_eps : float (optional)
            The eps value in layer normalization components. Defaults to 1e-5.
    batch_first : bool (optional)
        Whether the first dimension is the batch dimension. Defaults to False.
    norm_first : bool (optional)
            if True, layer norm is done prior to attention and feedforward. 
            Otherwise it’s done after. Defaults to False.
    device : str (optional)
            'cpu' or 'gpu' possible. Defaults to None.
    dtype _type_ : (optional): 
            Data type used for the input and output Tensor. Defaults to None.

    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None
    ) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PCTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        self.pc_self_attn = PCMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)

        # PC layers
        self.pc_norm_layer1 = PCLayer(size=d_model)
        self.pc_norm_layer2 = PCLayer(size=d_model)
        self.pc_ff_layer1 = PCLayer(size=dim_feedforward)
        self.pc_ff_layer2 = PCLayer(size=d_model)


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Pass the input through the encoder layer.
        
        Parameters:
        ----------
            src : Tensor
                    The sequence to the encoder layer.
            src_mask : Tensor (optional)
                    The mask for the src sequence. Defaults to None.
            src_key_padding_mask : Tensor (optional)
                    The mask for the src keys per batch. Defaults to None.
        
        Returns
        -------
        Returns the output of the encoder layer given the input sequence and masks.
        
        """

        x = src
        if self.norm_first:
            x = x + self._pc_sa_block(self.pc_norm_layer1(self.norm1(x)), src_mask, src_key_padding_mask)
            x = x + self._pc_ff_block(self.pc_norm_layer2(self.norm2(x)))
        else:
            x = self.pc_norm_layer1(self.norm1(x + self._pc_sa_block(x, src_mask, src_key_padding_mask)))
            x = self.pc_norm_layer2(self.norm2(x + self._pc_ff_block(x)))

        return x


    def _pc_sa_block(
        self, x: Tensor,
        attn_mask: Optional[Tensor], 
        key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        """
        PC Self-Attention Block
        
        """
        x = self.pc_self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        return self.dropout1(x)


    def _pc_ff_block(self, x: Tensor) -> Tensor:
        """
        PC Feed-Forward Block
        
        """
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.pc_ff_layer1(x)
        x = self.dropout2(self.linear2(x))
        x = self.pc_ff_layer2(x)
        return x
