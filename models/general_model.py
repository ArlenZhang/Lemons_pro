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
    def __init__(self):
        super(Lemon_Model, self).__init__()
        self.encoder_ = Encoder_()
        self.decoder_ = Decoder_()
        self.word_embed = nn.Embedding(WORD_LEN, EMBED_SIZE)
        self.pos_embed = nn.Embedding(POS_LEN, POS_EMB_SIZE)
        self.ent_embed = nn.Embedding(ENT_LEN, ENT_EMB_SIZE)
        self.dist_embed = nn.Embedding(DIST_LEN, DIST_EMB_SIZE)
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
        return self.word_embed(document.title_word_ids), self.pos_embed(document.title_pos_ids), self.ent_embed(
            document.title_ent_ids)

    def score_(self, output, gold_emb):
        return output, gold_emb

    def encode(self, word_embed, pos_embed, ent_embed):
        print(word_embed.size())
        print(pos_embed.size())
        print(ent_embed.size())
        doc_rep = torch.cat((word_embed, pos_embed, ent_embed), 0)
        input(doc_rep.size())
        return self.encoder_(doc_rep)

    def decode(self, encode_h, title_embed, title_len):
        """
        对ct逐个解码得到标题的过程
        :param title_len:
        :param title_embed:
        :param encode_h:
        :return:
        """
        decoded_output = None
        state_x_ = encode_h
        for idx in range(title_len):
            # 分析：
            #     这里需要根据距离差作为特征学习解码结束的标签位置。分析为什么不适用title_len而是使用MaxLen作
            #     为学习的距离目标。如果使用title_len那么模型将大比分看中dist==0的情况，而学不到差距越小越要
            #     结束这样的特性。使用Max_Len抓住了数据本身分布的特性，将大部分结束都控制在正态分布觉得最合理
            #     的距离差范围内。
            tmp_dist_emb = self.dist_embed(MAX_TITLE_LENGTH - (idx + 1))  # 这个值对于学习结束符很有帮助
            title_embed_ = title_embed[idx]
            input_embed = torch.cat((title_embed_, tmp_dist_emb), 1)  # 需要将标题单词向量和距离信息再做拼接
            output, state_x_ = self.decoder_(input_embed, state_x_)
            decoded_output = output if decoded_output is None else torch.cat((decoded_output, output), 0)
        score_out = self.score_(decoded_output, title_embed)
        return score_out

    def forward(self, document):
        """
        模型的编码解码打分结果
        :return:
        """
        word_embed, pos_embed, ent_embed = self.get_embedding_reps(document=document)
        title_word_embed, title_pos_embed, title_ent_embed = self.get_title_embedding_reps(document=document)
        title_rep = torch.cat((title_word_embed, title_pos_embed, title_ent_embed), 0)
        encode_h = self.encode(word_embed, pos_embed, ent_embed)
        input(encode_h.size())
        return self.decode(encode_h, title_rep, len(document.title_word_ids))
