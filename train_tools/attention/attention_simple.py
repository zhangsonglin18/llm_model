# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2024/1/19 14:29
# @Author  : changqingai
# @FileName: attention_simple.py
# ----------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        # Define the dimension of each head or subspace
        self.d_k = d_model // self.num_heads

        # These are still of dimension d_model. They will be split into number of heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        K_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesn't explode
        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply the mask
        if mask is not None:
            QK = QK.masked_fill(mask.to(QK.dtype) == 0, float('-inf'))

        # Calculate the attention weights (softmax over the last dimension)
        weights = F.softmax(QK, dim=-1)

        # Apply the self attention to the values
        attention = torch.matmul(weights, V)

        return attention, weights

    def split_heads(self, x, batch_size):
        """
        The original tensor with dimension batch_size * seq_length * d_model is split into num_heads
        so we now have batch_size * num_heads * seq_length * d_k
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # linear layers
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        # split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # self attention
        scores, weights = self.scaled_dot_product_attention(q, k, v, mask)

        # concatenate heads
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # final linear layer
        output = self.W_o(concat)

        return output, weights