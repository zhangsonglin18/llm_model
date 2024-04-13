# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2023/12/14 19:16
# @Author  : acedar
# @FileName: 01trainable_emb.py
# ----------------------------


from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import random
import numpy as np
import torch
from torch import nn
logger = logging.getLogger(__name__)


class TrainableEmbeddings(nn.Module):
    def __init__(self, config):
        super(TrainableEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        return embeddings


class Config(object):
    vocab_size = 10
    hidden_size = 8
    max_position_embeddings = 32
    type_vocab_size = 32
    hidden_dropout_prob = 0.2


if __name__ == "__main__":
    """PyTorch BERT 中的位置编码实现：可训练位置特征 """
    config = Config()
    bert_emb = TrainableEmbeddings(config)
    input_ids = [random.randint(0, config.vocab_size - 1) for i in range(16)]
    input_ids = torch.from_numpy(np.asarray([input_ids]))
    emb_res = bert_emb(input_ids)
    print("emb_res", emb_res)
