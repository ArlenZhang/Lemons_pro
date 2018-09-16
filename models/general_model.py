# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: 模型封装整个神经网络的参数信息本身，作为学习的目标
"""
import torch
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

    def get_text_embedding_reps(self, document):
        """
        从document对象中获取词，pos,ent等ids信息并转embedding表示返回
        :param document:
        :return:
        """
        return self.word_embed(document.text_word_ids), self.pos_embed(document.text_pos_ids), self.ent_embed(document.text_ent_ids)

    def get_title_embedding_reps(self, document):
        """
        获取标题embedding
        :param document:
        :return:
        """
        return self.word_embed(document.title_word_ids)

    def score_(self, output, gold_emb):
        return output, gold_emb

    def encode(self, word_embed, pos_embed, ent_embed):
        print(word_embed.size())
        print(pos_embed.size())
        print(ent_embed.size())
        doc_rep = torch.cat((word_embed, pos_embed, ent_embed), 0)
        input(doc_rep.size())
        return self.encoder_(doc_rep)

    def decode(self, encode_h, title_embed):
        """
        对ct逐个解码得到标题的过程
        :param title_embed:
        :param encode_h:
        :return:
        """
        decoded_output = None
        state_x_ = encode_h
        for idx in range(TITLE_LENGTH):
            title_embed_ = title_embed[idx]
            output, state_x_ = self.decoder_(title_embed_, state_x_)
            decoded_output = output if decoded_output is None else torch.cat((decoded_output, output), 0)
        score_out = self.score_(decoded_output, title_embed)
        return score_out

    def forward(self, document):
        """
        模型的编码解码打分结果
        :return:
        """
        word_embed, pos_embed, ent_embed = self.get_embedding_reps(document=document)
        title_embed = self.get_title_embedding_reps(document=document)
        encode_h = self.encode(word_embed, pos_embed, ent_embed)
        input(encode_h.size())
        return self.decode(encode_h, title_embed)
