# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: Decode those hidden states into title.
"""
import torch.nn as nn
from config import *


class Decoder_(nn.Module):
    def __init__(self):
        super(Decoder_, self).__init__()
        self.lstm = nn.LSTMCell(EMBED_SIZE, HIDDEN_SIZE)
        self.out = nn.Linear(HIDDEN_SIZE, DECODE_OUT_SIZE)  # 将隐层输出维度和词表大小保持一致

    def forward(self, input_=None, state_x=None):
        """

        :param state_x: (80, 1000)
        :param input_: 目标端句子 (80, 620)
        :return:
        """
        hidden_x_, cell_x_ = self.lstm(input_, state_x)  # input_为目标端句子向量表示，state_x初始为编码端的输出，随后迭代
        output = self.out(hidden_x_)
        state_x_ = (hidden_x_, cell_x_)
        return output, state_x_
