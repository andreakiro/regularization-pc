# PC https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
from typing import Union, Callable, Optional

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import TransformerEncoderLayer
from torch.nn.modules.transformer import _get_activation_fn

from src.layers import PCLayer, PCMultiheadAttention

class PCTransformerEncoderLayer(TransformerEncoderLayer):
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
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x = x + self._pc_sa_block(self.pc_norm_layer1(self.norm1(x)), src_mask, src_key_padding_mask)
            x = x + self._pc_ff_block(self.pc_norm_layer2(self.norm2(x)))
        else:
            x = self.pc_norm_layer1(self.norm1(x + self._pc_sa_block(x, src_mask, src_key_padding_mask)))
            x = self.pc_norm_layer2(self.norm2(x + self._pc_ff_block(x)))

        return x

    # PC self-attention block
    def _pc_sa_block(
        self, x: Tensor,
        attn_mask: Optional[Tensor], 
        key_padding_mask: Optional[Tensor]
    ) -> Tensor:
    
        x = self.pc_self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        return self.dropout1(x)

    # PC feed forward block
    def _pc_ff_block(self, x: Tensor) -> Tensor:
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.pc_ff_layer1(x)
        x = self.dropout2(self.linear2(x))
        x = self.pc_ff_layer2(x)
        return x
