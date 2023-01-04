from torch import Tensor
from torch.nn import LayerNorm
from torch.nn.modules.transformer import Transformer
from torch.nn.modules.transformer import TransformerEncoder
import torch.functional as F

from typing import Union, Callable, Optional, Any

from src.layers import PCTransformerEncoderLayer


class PCTransformer(Transformer):
    r"""A PC transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
        layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
        device=None, dtype=None
    ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        pc_encoder_layer = PCTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps, batch_first, norm_first,
            **factory_kwargs
        )

        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        custom_encoder = TransformerEncoder(
            pc_encoder_layer, num_encoder_layers, encoder_norm)

        super(PCTransformer, self).__init__(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers, dim_feedforward, dropout,
            activation,
            custom_encoder,
            custom_decoder,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype
        )
