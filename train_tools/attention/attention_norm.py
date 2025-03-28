# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2024/1/19 14:29
# @Author  : changqingai
# @FileName: attention_norm.py
# ----------------------------


import torch
from torch.nn import Linear
import math


class Attention(torch.nn.Module):  # 这里bmt.DistributedModule你不用在意，简单理解为torch.nn.Module即可

    """ Attention module consisting procedure of Q, K, V combination and its output projection.
    For more detail, see `Attention is All you Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        dim_in (int): input dimension.
        dim_head (int): dimension of each heads used in attention.
        num_heads (int): number of heads used in attention.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attetion module. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attention module. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in attention module. Defaults to False.
        mask_value (float, optional): mask value of the masked position. Defaults to `-inf`.
        pos_bias_type (str, optional): `relative` for relative position bias, `rotary` for ratery position embedding. Defaults to `none`.
        attn_scale (bool, optional): whether to scale before softmax, i.e., :math:`\text{softmax}({Q K^T \over \sqrt{\text{dim_model}}})`. Default to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_in: int,
                 dim_head: int,
                 num_heads: int,
                 dim_out: int = None,
                 dtype=torch.half,
                 int8=False,
                 init_mean=0.0,
                 init_std=0.02,
                 bias=False,
                 mask_value: float = float("-inf"),
                 pos_bias_type: str = "none",
                 length_scale: bool = False,
                 attn_scale: bool = False,
                 dropout_p: float = 0,
                 shared_key_and_value=False,
                 ):

        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        num_heads_kv = 1 if shared_key_and_value else num_heads  # 这里可以选择Multi-Query Attention(MQA)，MHA/MQA/GQA的对比可以看https://zhuanlan.zhihu.com/p/644698212

        # 下面是四个最重要的线性层project_q,project_k,project_v,attention_out
        # 注意这里矩阵的输出维度，有高并行的优点。除了输入输出外其他的一些参数继承自线性层的实现即可。
        self.project_q = Linear(
            dim_in=dim_in,
            dim_out=num_heads * dim_head,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        # project_k和project_v的头数是num_heads_kv
        self.project_k = Linear(
            dim_in=dim_in,
            dim_out=num_heads_kv * dim_head,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        self.project_v = Linear(
            dim_in=dim_in,
            dim_out=num_heads_kv * dim_head,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        # 将多头转换回输出维度
        self.attention_out = Linear(
            dim_in=num_heads * dim_head,
            dim_out=dim_out,
            length_scale=length_scale,
            length_scale_before=False,
            dtype=dtype,
            int8=int8,
            init_mean=init_mean,
            init_std=init_std,
            bias=bias,
        )

        self.init_mean = init_mean
        self.init_std = init_std
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.dim_out = dim_out
        self.int8 = int8
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.mask_value = mask_value
        self.dtype = dtype
        self.dropout_p = dropout_p
        self.shared_key_and_value = shared_key_and_value

        if dropout_p:
            self.attention_dropout = torch.nn.Dropout(dropout_p)
        else:
            self.attention_dropout = None

        self.bias = bias
        self.pos_bias_type = pos_bias_type
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor,
                key_value: torch.Tensor,
                attention_mask: torch.Tensor,
                position_bias: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_value=None,
                ):

        """ This model inherits from bmt.DistributedModule.

        Args:
            query (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            key_value (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        h_q = self.project_q(query)  # (batch, len_q, num_heads * dim_head)
        h_k = self.project_k(key_value)  # (batch, len_k, num_heads * dim_head)
        h_v = self.project_v(key_value)  # (batch, len_k, num_heads * dim_head)

        # 拆头
        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1,
                                                                                 3)  # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1,
                                                                                    3)  # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1,
                                                                                    3)  # (batch, num_heads_kv, len_k, dim_head)

        # if self.shared_key_and_value:
        #     h_k = h_k.repeat(1, self.num_heads, 1, 1)
        #     h_v = h_v.repeat(1, self.num_heads, 1, 1)

        h_q = h_q.contiguous()  # (batch, num_heads, len_q, dim_head)
        h_k = h_k.contiguous()  # (batch, num_heads, len_k, dim_head)
        h_v = h_v.contiguous()  # (batch, num_heads, len_k, dim_head)

        # 自回归常用的优化trick，decoding到第t步的时候，前t-1步已经计算了key和value，所以保存下来，避免重复计算。
        # encoding不需要，因为输入是固定的，一个step就可以，不需要recursive地生成。这是auto-regressive特有的trick
        if past_key_value is not None:
            h_k = torch.cat([past_key_value[0], h_k], dim=-2)
            h_v = torch.cat([past_key_value[1], h_v], dim=-2)
            len_k = h_k.size(-2)

        current_key_value = (h_k, h_v) if use_cache else None

        # 如果模型采用RoPE位置编码的话，在这里要为h_q, h_k赋予位置信息
        if self.pos_bias_type == "rotary":
            h_q, h_k = position_bias(h_q, h_k)

        # (batch, num_heads, len_q, dim_head) @ (batch, num_heads_kv, len_k, dim_head)T
        # => (batch, num_heads, len_q, len_k)

        # 算Attn score
        score = torch.matmul(h_q, h_k.transpose(2, 3))
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        # (batch, num_heads, len_q, len_k)
        # score = score.view(batch_size, self.num_heads, len_q, len_k)

        # 其他相对位置编码直接加在Attn score上
        if self.pos_bias_type == "relative":
            if position_bias is not None:
                # (batch, num_heads, len_q, len_k) + (1, num_heads, len_q, len_k)
                score = score + position_bias

        # 对score填充mask，第二个参数矩阵中True表示要填充，attention_mask本身非0表示有效
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,
            torch.scalar_tensor(self.mask_value, device=score.device, dtype=score.dtype)
        )  # (batch, num_heads, len_q, len_k)

        # 过softmax
        score = self.softmax(score)

        # avoid nan in softmax，一些数值稳定相关的问题
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )
        # .view(batch_size * self.num_heads, len_q, len_k) # (batch * num_heads, len_q, len_k)

        # 如果需要，加dropout
        if self.attention_dropout is not None:
            score = self.attention_dropout(score)

        # (batch * num_heads, len_q, len_k) @ (batch * num_heads, len_k, dim_head) = (batch * num_heads, len_q, dim_head)
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1,
                                                                                     3)  # (batch, len_q, num_heads, dim_head)
        score = score.reshape(batch_size, len_q, self.num_heads * self.dim_head)  # (batch, len_q, num_heads * dim_head)

        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        score = self.attention_out(score)

        # 还是decoding的时候是否使用past_key_value的策略。
        if use_cache:
            return score, current_key_value
        else:
            return score

