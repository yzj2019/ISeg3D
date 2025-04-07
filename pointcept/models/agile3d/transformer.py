"""
部分借鉴Meta

transformer layer/block, 以及最基本的 Transformer decoder 示例
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import warnings
from typing import Tuple, Type

from .misc import MLPBlock

from .attention import FlashMHA


# ------------------------------------------------------------------------
# 改编自 Segment Anything
# 双通道 transformer block & decoder, 互相查询, 用的是 FlashMHA, 算是个示例
# ------------------------------------------------------------------------
class FlashTwoWayTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        attn_drop: int = 0.1,
        layer_drop: int = 0.1,
        norm_first=False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
        - embedding_dim (int): the channel dimension of the embeddings
        - num_heads (int): the number of heads in the attention layers
        - mlp_dim (int): the hidden dimension of the mlp block
        - activation (nn.Module): the activation of the mlp block
        - skip_first_layer_pe (bool): skip the PE on the first layer
        - attn_drop: dropout percent in each attn layer, default 0.1
        - layer_drop: dropout percent in each dropout layer, default 0.1
        - nrom_first: whether to normalize before attn (unused)
        """
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe
        self.self_attn = FlashMHA(embedding_dim, num_heads, dropout=attn_drop)
        self.drop1 = nn.Dropout(layer_drop)
        self.norm1 = nn.LayerNorm(
            embedding_dim
        )  # 可合并成 attention.py 里的 SelfAttentionLayer

        self.cross_attn_token_to_image = FlashMHA(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            dropout=attn_drop,
        )
        self.drop2 = nn.Dropout(layer_drop)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.drop3 = nn.Dropout(layer_drop)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = FlashMHA(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.drop4 = nn.Dropout(layer_drop)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
            queries = self.drop1(queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + self.drop1(attn_out)
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + self.drop2(attn_out)
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + self.drop3(mlp_out)
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + self.drop4(attn_out)
        keys = self.norm4(keys)

        return queries, keys


class FlashTwoWayTransformerDecoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                FlashTwoWayTransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = FlashMHA(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
