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
        self.dist_embed.requires_grad = True

    def sent2vec(self, text_word, text_pos, text_ent):
        """
        输入一个篇章的三类数据的位置信息
        掉用pca方案对句子进行编码，返回一个篇章所有句子的embedding
        :return:
        """
        return 1

    def get_text_embedding_reps(self, text_ids):
        """

        :param text_ids: 一批数据的合集 (batch_size, sent_num, sent_len)
        :return: batch_sent_list (batch_size, sent_num, sent_embedding_size)
        """
        batch_sent_list = []
        batch_text_word_ids, batch_text_pos_ids, batch_text_ent_ids = text_ids
        for text_word, text_pos, text_ent in zip(batch_text_word_ids, batch_text_pos_ids, batch_text_ent_ids):
            sent_list = self.sent2vec(text_word, text_pos, text_ent)
            batch_sent_list.append(sent_list)
        return batch_sent_list

    def encode(self, batch_sent_list):
        return self.encoder_(batch_sent_list)

    def decode(self, encode_h, title_embed):
        """
        对ct逐个解码得到标题的过程
        :param title_embed:
        :param encode_h:
        :return:
        """
        batch_out = None
        state_x_ = encode_h
        for idx in range(MAX_TITLE_LENGTH):
            # 分析：
            #     这里需要根据距离差作为特征学习解码结束的标签位置。分析为什么不适用title_len而是使用MaxLen作
            #     为学习的距离目标。如果使用title_len那么模型将大比分看中dist==0的情况，而学不到差距越小越要
            #     结束这样的特性。使用Max_Len抓住了数据本身分布的特性，将大部分结束都控制在正态分布觉得最合理
            #     的距离差范围内。
            tmp_dist_emb = self.dist_embed(MAX_TITLE_LENGTH - (idx + 1))  # 这个值对于学习结束符有帮助
            title_word_embed_ = title_embed[idx]
            input_embed = torch.cat((title_word_embed_, tmp_dist_emb), 1)  # 需要将标题单词向量和距离信息再做拼接
            output, state_x_ = self.decoder_(input_embed, state_x_)
            # 对输出进行拼接
            batch_out = output if batch_out is None else torch.cat((batch_out, output), 1)
        return batch_out

    def title_generate(self, all_ids_test):
        """
        根据input直接生成输出信息
        all_ids_test : 测试集中的关于文本内容部分的ids情况
        :return:
        """
        text_word_batch, text_pos_batch, text_ent_batch = all_ids_test
        batch_sent_list = self.get_text_embedding_reps(text_ids=(text_word_batch, text_pos_batch, text_ent_batch))
        encode_h = self.encode(batch_sent_list)
        # 这里采用特殊的解码方式进行解码：初始化一个词ids为pad以词向量信息作为y_0输入，根据encode_h和y_0解码出y_1和h_1一次类推
        # ，直到最大标题长度为止 rnn使用 self.decoder_.lstm
        trg_ = torch.zeros((len(text_word_batch), EMBED_SIZE))
        out_concat = None
        for idx in MAX_TITLE_LENGTH:
            output, state_x = self.decoder(trg_, state_x)
            output_ids = self.decode2trg_ids(output=output)
            trg_ = self.word_embed(output_ids)
            out_concat = torch.unsqueeze(output_ids, 0) if out_concat is None \
                else torch.cat((out_concat, torch.unsqueeze(output_ids, 0)),
                               dim=0)
            if output_ids == END_ID:
                # 预测到结束符则直接结束
                return out_concat
        return out_concat

    def decode2trg_ids(output):
        """
        将output概率化（softmax），找到概率最大的词语对应的下标，从而转换为词语的id表示(80,1)
        :param output: 每一轮lstmcell的输出:（80，30001）  BATCH_SIZE:80, TRG_VOCAB_SIZE:30001
        :return: BATCH个句子的所有第一个词语的下标
        """
        softmax_pro_out = self.softmax(output)
        trg_ids = torch.max(softmax_pro_out, dim=1)  # 返回一个2元组，[0]为最大概率，[1]为最大概率对应的下标
        return trg_ids[1]

    def forward(self, all_ids):
        """
        模型的编码解码打分结果
        输入描述：all_ids = [batch_size, sent_num, sent_len]
        :return:
        """
        text_word_batch, text_pos_batch, text_ent_batch, title_word_batch = all_ids
        # 下面将(batch_size, sent_num, sent_len) 转成 (batch_size, sent_num, sent_embedding_size)的形式
        batch_sent_list = self.get_text_embedding_reps(text_ids=(text_word_batch, text_pos_batch, text_ent_batch))
        # shape (batch_size, hidden_size)
        encode_h = self.encode(batch_sent_list)
        input(encode_h.size())
        # (batch_size, title length, embed_size)
        title_word_embed = self.word_embed(title_word_batch)
        return self.decode(encode_h, title_word_embed)
