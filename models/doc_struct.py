# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: 封装篇章结构体
"""


class Document:
    def __init__(self):
        # 针对内容
        self.text_word_ids = ...
        self.text_pos_ids = ...
        self.text_ent_ids = ...
        # 针对标题
        self.title_word_ids = ...
        self.title_pos_ids = ...
        self.title_ent_ids = ...
