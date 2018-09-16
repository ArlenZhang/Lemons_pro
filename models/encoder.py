# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: Encode those documents into hidden states.
"""
import torch
import torch.nn as nn
from config import *
from torch.nn import functional as nnfunc


class Encoder_(nn.Module):
    def __init__(self):
        super(Encoder_, self).__init__()
        # self-attention
        self.edu_rnn_encoder = nn.LSTM(EMBED_SIZE + POS_EMB_SIZE + ENT_EMB_SIZE, HIDDEN_SIZE, bidirectional=True)
        self.edu_attn_query = nn.Parameter(torch.randn(HIDDEN_SIZE))
        self.edu_attn = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh()
        )

    def bilstm_attn_encode(self, doc_rep):
        """
        对 edu 双向 lstm 进行编码
        :param doc_rep:
        :return:
        """
        inputs = doc_rep.unsqueeze(1)  # (seq_len, batch, input_size)
        hs, _ = self.edu_rnn_encoder(inputs)  # hs.size()  (seq_len, batch, hidden_size)
        hs = hs.squeeze()  # size: (seq_len, hidden_size)
        keys = self.edu_attn(hs)  # size: (seq_len, hidden_size)
        attn = nnfunc.softmax(keys.matmul(self.edu_attn_query), 0)
        output = (hs * attn.view(-1, 1)).sum(0)
        return output, attn

    def forward(self, doc_rep=None):
        """

        :param doc_rep: 包含text_embedding, pos, ent的embedding表示 (n_words, all_embed_size)
        :return:
        """
        return self.bilstm_attn_encode(doc_rep=doc_rep)
