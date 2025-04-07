"""
注意力机制的计算, 以及封装成 attention layer/block

默认 batch_first=True
仿照 nn.MultiheadAttention 和 F.multi_head_attention_forward, 只写 attn_drop; 
layer norm、attention和ffn linear/activation 后的drop out、shortcut、positional embedding, 都写在 transformer layer里;
注意在 MHA 中实现 attn_mask、key_padding_mask
"""

import torch
import torch.nn as nn

# Flash atten
# from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
# from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis

from einops import rearrange
import math

# from mmcv.runner import auto_fp16

from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.functional import linear


# -------------------------------------------------------------------------
# Flash attn
# 官方没实现 attn mask, 只有 key padding mask, 用作筛选kv
# -------------------------------------------------------------------------
# class FlashAttention(nn.Module):
#     """attention计算的封装实现, 没做 Multihead:
#     Implement the scaled dot product attention with softmax.
#     在 flash_attn_unpadded_kvpacked_func 里封装了对 sqrt head_dim 的归一化, 以及softmax, 但要指定 softmax_scale
#     """
#     def __init__(self, softmax_scale=None, attention_dropout=0., device=None, dtype=None):
#         '''
#         Arguments
#         ---------
#         softmax_scale: The temperature to use for the softmax attention.
#                       (default: 1/sqrt(d_keys) where d_keys is computed at
#                       runtime)
#         attention_dropout: The dropout rate to apply to the attention
#                            (default: 0.1)
#         即使加了 key_padding_mask, flash_attn_unpadded_kvpacked_func 函数也会填充回原shape
#         '''
#         super().__init__()
#         self.softmax_scale = softmax_scale
#         self.dropout_p = attention_dropout
#         self.fp16_enabled = True

#     # 没用混合精度的hooker，用什么代替？autocast
#     # @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
#     def forward(self, q, kv,
#                 causal=False,
#                 key_padding_mask=None):
#         """Implements the multihead softmax attention.
#         Arguments
#         ---------
#             q: The tensor containing the query. (B, T, H, D) , (batch_size, tgt_seqlen, nheads, headdim)
#             kv: The tensor containing the key, and value. (B, S, 2, H, D)
#             key_padding_mask: a bool tensor of shape (B, S), indicating which elements within `kv` to ignore for the purpose of attention
#         """
#         assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
#         assert q.is_cuda and kv.is_cuda
#         assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

#         batch_size = q.shape[0]
#         seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
#         if key_padding_mask is None:
#             q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
#             max_sq, max_sk = seqlen_q, seqlen_k
#             cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
#                                     device=q.device)
#             cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
#                                     device=kv.device)
#             output = flash_attn_unpadded_kvpacked_func(
#                 q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
#                 self.dropout_p if self.training else 0.0,
#                 softmax_scale=self.softmax_scale, causal=causal
#             )
#             output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
#         else:
#             nheads = kv.shape[-2]
#             q = rearrange(q, 'b s ... -> (b s) ...')
#             max_sq = seqlen_q
#             cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
#                                     device=q.device)
#             x = rearrange(kv, 'b s two h d -> b s (two h d)')
#             x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
#             x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
#             # 如果 model.eval(), 则子模块 training 设置为false
#             output_unpad = flash_attn_unpadded_kvpacked_func(
#                 q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
#                 self.dropout_p if self.training else 0.0,
#                 softmax_scale=self.softmax_scale, causal=causal
#             )
#             output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

#         return output, None


# def _in_projection(q, k, v, w_q, w_k, w_v, b = None):
#     '''用分开的qkv做输入时的proj'''
#     if b is None:
#         b_q = b_k = b_v = None
#     else:
#         b_q, b_k, b_v = b.chunk(3)
#     return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

# def _in_projection_packed(q, k, v, w, b = None):
#     '''用packed w,b 做输入时qkv的proj'''
#     w_q, w_k, w_v = w.chunk(3)          # 在dim=0拆分tensor
#     if b is None:
#         b_q = b_k = b_v = None
#     else:
#         b_q, b_k, b_v = b.chunk(3)
#     return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# class FlashMHA(nn.Module):
#     '''flash attention实现的multi head attention, 仿照 `torch.nn.MultiheadAttention`'''
#     def __init__(self,
#                  embed_dim,
#                  num_heads,
#                  downsample_rate=1,
#                  bias=True,
#                  batch_first=True,
#                  dropout=0., causal=False,
#                  kdim=None, vdim=None,
#                  device=None, dtype=None, **kwargs) -> None:
#         '''Arguments:
#         - downsample_rate: embed_dim 除 internal embed dim, 压缩维度省attn内存
#         - batch_first: whether
#         - dropout: Dropout probability on `attn_output_weights`
#         '''
#         assert batch_first, "must be batch first"
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         # qkv的embed dim
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
#         # internal dim, 压缩维度后的
#         assert embed_dim % downsample_rate == 0, "embed_dim must be divisible by downsample_rate"
#         self.internal_dim = embed_dim // downsample_rate
#         self.causal = causal
#         self.bias = bias
#         # head dim
#         self.num_heads = num_heads
#         assert self.internal_dim % num_heads == 0, "self.internal_dim must be divisible by num_heads"
#         self.head_dim = self.internal_dim // num_heads
#         assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

#         # 参数注册
#         if self._qkv_same_embed_dim is False:
#             self.q_proj_weight = nn.Parameter(torch.empty((self.internal_dim, embed_dim), **factory_kwargs))
#             self.k_proj_weight = nn.Parameter(torch.empty((self.internal_dim, self.kdim), **factory_kwargs))
#             self.v_proj_weight = nn.Parameter(torch.empty((self.internal_dim, self.vdim), **factory_kwargs))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = nn.Parameter(torch.empty((3 * self.internal_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)
#         if bias:
#             self.in_proj_bias = nn.Parameter(torch.empty(3 * self.internal_dim))
#         else:
#             self.register_parameter('in_proj_bias', None)

#         # 与pytorch的不同点在于这里把attention作为类写进来了, 而不是以function的形式
#         self.inner_attn = FlashAttention(softmax_scale=1/math.sqrt(self.head_dim), attention_dropout=dropout, **factory_kwargs)
#         self.out_proj = nn.Linear(self.internal_dim, embed_dim, bias=bias)
#         # if add_bias_kv:
#         #     self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         #     self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         # else:
#         #     self.bias_k = self.bias_v = None

#         # self.add_zero_attn = add_zero_attn
#         self._reset_parameters()

#     def _reset_parameters(self) -> None:
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)

#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         # if self.bias_k is not None:
#         #     xavier_normal_(self.bias_k)
#         # if self.bias_v is not None:
#         #     xavier_normal_(self.bias_v)

#     # 作为一个允许不同 ckpt 的示例
#     # def __setstate__(self, state):
#     #     # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#     #     if '_qkv_same_embed_dim' not in state:
#     #         state['_qkv_same_embed_dim'] = True

#     #     super(MultiheadAttention, self).__setstate__(state)

#     def forward(self, q, k, v, key_padding_mask=None):
#         """
#         where internal_dim = num heads * head dim, embed_dim = self.internal_dim * downsample_rate
#         - q: (batch, tgt_seqlen, embed_dim), if not batched, unsqueeze(0) before input
#         - k: (batch, src_seqlen, kdim)
#         - v: (batch, src_seqlen, vdim)
#         - key_padding_mask: bool tensor of shape (batch, src_seqlen), indicating which elements within `kv` to ignore for the purpose of attention
#         """
#         assert q.dim()==3 and k.dim()==3 and v.dim()==3, "check qkv dim == 3"
#         #
#         # compute in-projection, 将 embed dim 投影到一致
#         if self._qkv_same_embed_dim:
#             q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
#         else:
#             q, k, v = _in_projection(q, k, v, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, self.in_proj_bias)

#         #
#         # rearrange (.contiguous().view()); reshape q, k, v for multihead attention
#         q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
#         k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
#         v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
#         kv = torch.stack([k, v], dim=2)         # b s 2 h d

#         #
#         # attention & out projection
#         context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
#         return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


# -------------------------------------------------------------------------
# 普通的 attn, from https://github.com/sunjiahao1999/SPFormer
# 参考了 Mask3D, 但改成了 squeezed batch 版本: 对不定长 q k, 只能在自己 batch 内查询, 不然attn会互相影响, 因为做了softmax
# -------------------------------------------------------------------------
class SqueezedCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, layer_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(layer_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        source,
        query,
        source_batch,
        query_batch,
        attn_mask=None,
        source_pe=None,
        query_pe=None,
    ):
        """
        - source: Tensor (n_s, embed_dim)
        - source_batch: (n_s), which batch the source belongs to
        - query: Tensor (n_q, embed_dim)
        - query_batch: (n_q), which batch the query belongs to
        - attn_mask: (n_q, n_s)
        - source_pe: (n_s, embed_dim)
        - query_pe: (n_q, embed_dim)
        - output: (n_q, embed_dim)
        """
        assert (
            query_batch.max() <= source_batch.max()
        ), "query_batch.max() > source_batch.max()"
        B = query_batch.max() + 1
        output = torch.empty_like(query, dtype=query.dtype, device=query.device)
        source_with_pe = self.with_pos_embed(source, source_pe)
        query_with_pe = self.with_pos_embed(query, query_pe)
        for b in range(B):
            source_batch_mask = source_batch == b
            query_batch_mask = query_batch == b
            q = query_with_pe[query_batch_mask].unsqueeze(0)
            k = source_with_pe[source_batch_mask].unsqueeze(0)  # (1, n_s_b, embed_dim)
            v = source[source_batch_mask].unsqueeze(0)
            if attn_mask != None:
                attn_mask_b = attn_mask[query_batch_mask]  # (n_q_b, n_s)
                attn_mask_b = attn_mask_b[:, source_batch_mask]  # (n_q_b, n_s_b)
                attn_out, _ = self.attn(
                    q, k, v, attn_mask=attn_mask_b
                )  # (1, n_q_b, embed_dim)
            else:
                attn_out, _ = self.attn(q, k, v)
            attn_out = (
                self.dropout(attn_out).squeeze(0) + query[query_batch_mask]
            )  # (n_q_b, embed_dim)
            attn_out = self.norm(attn_out)
            output[query_batch_mask] = attn_out  # 索引赋值时会正确处理梯度？
        return output


class SqueezedSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, layer_drop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(layer_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_batch, pe=None):
        """
        - x: Tensor (n_x, embed_dim)
        - x_batch: (n_x), which batch x belongs to
        - pe: Tensor (n_x, embed_dim)
        - output: Tensor (n_x, embed_dim)
        """
        B = x_batch.max() + 1
        output = torch.empty_like(x, dtype=x.dtype, device=x.device)
        x_with_pe = self.with_pos_embed(x, pe)
        for b in range(B):
            batch_mask = x_batch == b
            q = k = x_with_pe[batch_mask].unsqueeze(0)
            v = x[batch_mask].unsqueeze(0)
            attn_out, _ = self.attn(q, k, v)
            attn_out = self.dropout(attn_out) + v
            attn_out = self.norm(attn_out)
            output[batch_mask] = attn_out.squeeze(0)
        return output
