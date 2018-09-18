# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: 封装篇章结构体
"""


class Document:
    def __init__(self):
        # 针对内容
        self.text_word_ids = None
        self.text_pos_ids = None
        self.text_ent_ids = None
        self.text_tokens = None

        # 针对标题
        self.title_word_ids = None
        self.title_tokens = None

        # 分句结果
        self.text_sents_ids = None
        self.text_sents_pos = None
        self.text_sents_ent = None
        self.text_sents_tokens = None

    def get_all(self):
        """
        返回所需数据
        :return:
        """
        return self.text_sents_ids, self.text_sents_pos, self.text_sents_ent, self.title_word_ids
