# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: 对数据进行迭代
"""
import random
from config import *
from utils.file_util import *


class data_iterator:
    def __init__(self):
        # 加载数据
        self.train_dt = load_data(TRAIN_documents)
        self.dev_dt = load_data(DEV_documents)
        # batch
        self.text_word_ids_batch = self.text_pos_ids_batch = self.text_ent_ids_batch = self.title_word_ids_batch = \
            self.title_pos_ids_batch = self.title_ent_ids_batch = None
        self.init_batch()
        random.seed(RANDOM_SEED)

    def init_batch(self):
        """
        对当前批次的数据进行初始化
        :return:
        """
        self.text_word_ids_batch = []
        self.text_pos_ids_batch = []
        self.text_ent_ids_batch = []
        self.title_word_ids_batch = []
        self.title_pos_ids_batch = []
        self.title_ent_ids_batch = []

    def shuffle_all(self):
        random.shuffle(self.train_dt)
        random.shuffle(self.dev_dt)

    def next_train_document(self):
        """
        返回下一个篇章的相关
        :return:
        """
        idx = 0
        while True:
            text_word_ids, text_pos_ids, text_ent_ids, title_word_ids, title_pos_ids, title_ent_ids, \
                title_tokens = self.train_dt[idx].get_all()
            idx += 1
            yield text_word_ids, text_pos_ids, text_ent_ids, title_word_ids, title_pos_ids, title_ent_ids, \
                title_tokens

    def next_train_batch(self):
        """
        返回下一批数据：内容的word_ids，pos，ent，对应标题的word_ids, pos, ent
        :return:
        """
        count = 0
        doc_ite = self.next_train_document()
        while True:
            text_word_ids, text_pos_ids, text_ent_ids, title_word_ids, title_pos_ids, title_ent_ids, title_tokens = \
                next(doc_ite)
            self.text_word_ids_batch.append(text_word_ids)
            self.text_pos_ids_batch.append(text_pos_ids)
            self.text_ent_ids_batch.append(text_ent_ids)
            self.title_word_ids_batch.append(title_word_ids)
            self.title_pos_ids_batch.append(title_pos_ids)
            self.title_ent_ids_batch.append(title_ent_ids)
            count += 1
            if count == BATCH_SIZE:
                count = 0
                yield self.text_word_ids_batch, self.text_pos_ids_batch, self.text_ent_ids_batch, \
                    self.title_word_ids_batch, self.title_pos_ids_batch, self.title_ent_ids_batch
                self.init_batch()
