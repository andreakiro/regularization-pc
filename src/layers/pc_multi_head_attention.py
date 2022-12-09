from torch import Tensor
from torch.nn import MultiheadAttention
from typing import Optional, Tuple

# To be completed @Franz
class PCMultiheadAttention(MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None, need_weights: bool = True, attn_mask: Optional[Tensor] = None, average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        return super().forward(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights)
