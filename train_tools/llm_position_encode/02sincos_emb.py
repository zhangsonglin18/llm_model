# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2023/12/14 20:30
# @Author  : acedar
# @FileName: 02sincos_emb.py
# ----------------------------


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
import random

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # exp(log(x)) = x
        # div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        print("position:", position.size())
        print("div_term:", div_term.size())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        print("pe:", pe.size())
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":
    """ transformer 的正弦位置编码"""
    d_model = 512
    dropout = 0.2
    vocab = 10
    position = PositionalEncoding(d_model, dropout)
    model = nn.Sequential(Embeddings(d_model, vocab), position)
    input_ids = [random.randint(0, vocab - 1) for i in range(16)]
    input_ids = torch.from_numpy(np.asarray([input_ids]))
    output = model(input_ids)
    print("output:", output)
