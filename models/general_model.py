# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: 模型封装整个神经网络的参数信息本身，作为学习的目标
"""
import torch.nn as nn
from config import *
from models.encoder import Encoder_
from models.decoder import Decoder_


class Lemon_Model(nn.Module):
    def __init__(self, word_len=None, pos_len=None, ent_len=None):
        super(Lemon_Model, self).__init__()
        self.encoder_ = Encoder_()
        self.decoder_ = Decoder_()
        self.word_embed = nn.Embedding(word_len, EMBED_SIZE)
        self.pos_embed = nn.Embedding(pos_len, POS_EMB_SIZE)
        self.ent_embed = nn.Embedding(ent_len, ENT_EMB_SIZE)
        self.wordemb.requires_grad = True
        self.pos_embed.requires_grad = True
        self.ent_embed.requires_grad = True

    def get_embedding_reps(self, document):
        """
        从document对象中获取词，pos,ent等ids信息并转embedding表示返回
        :param document:
        :return:
        """
        return self.word_embed(document.text_word_ids), self.pos_embed(document.text_pos_ids), self.ent_embed(document.text_ent_ids)

    def encode(self, doc_rep):
        return self.encoder_(doc_rep)

    def score(self, encode_h):
        return self.decoder_(encode_h)

    def forward(self, document):
        """
        模型的编码解码打分结果
        :return:
        """
        encode_h = self.encode(self.get_embedding_reps(document=document))
        return self.score(encode_h)
