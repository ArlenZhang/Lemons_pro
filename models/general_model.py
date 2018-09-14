# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description:
"""
import torch.nn as nn
from models.encoder import Encoder_
from models.decoder import Decoder_


class Lemon_Model(nn.Module):
    def __init__(self):
        super(Lemon_Model, self).__init__()
        self.encoder_ = Encoder_()
        self.decoder_ = Decoder_()

    def score(self):
        ...

    def forward(self):
        ...
