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
        self.train_file_paths = os.listdir(TRAIN_documents)
        self.tmp_set_idx = 0
        self.train_dt = load_data(self.train_file_paths[self.tmp_set_idx])
        self.dev_dt = load_data(DEV_documents)
        self.test_dt = load_data(TEST_documents)
        # batch
        self.text_word_ids_batch = self.text_pos_ids_batch = self.text_ent_ids_batch = self.title_word_ids_batch = None
        self.init_batch()
        self.doc_idx = 0
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

    def new_epoch(self):
        # 跟新self.train_dt用的数据集
        self.tmp_set_idx += 1
        self.train_dt = load_data(self.train_file_paths[self.tmp_set_idx % TRAIN_SET_NUM])
        random.shuffle(self.train_dt)
        self.doc_idx = 0

    def next_train_document(self):
        """
        返回下一个篇章的相关
        :return:
        """
        while True:
            text_sents_ids, text_sents_pos, text_sents_ent, title_word_ids = self.train_dt[self.doc_idx].get_all()
            self.doc_idx += 1
            yield text_sents_ids, text_sents_pos, text_sents_ent, title_word_ids

    def next_train_batch(self):
        """
        返回下一批数据：内容的word_ids，pos，ent，对应标题的word_ids, pos, ent
        :return:
        """
        count = 0
        doc_ite = self.next_train_document()
        while True:
            text_sents_ids, text_sents_pos, text_sents_ent, title_word_ids = next(doc_ite)
            self.text_word_ids_batch.append(text_sents_ids)
            self.text_pos_ids_batch.append(text_sents_pos)
            self.text_ent_ids_batch.append(text_sents_ent)
            self.title_word_ids_batch.append(title_word_ids)
            count += 1
            if count == BATCH_SIZE:
                count = 0
                yield self.text_word_ids_batch, self.text_pos_ids_batch, \
                    self.text_ent_ids_batch, self.title_word_ids_batch
                self.init_batch()

    def get_dev_all(self):
        """
        获取7号文件的前1000篇作为开发集
        :return:
        """
        text_word_ids_batch = []
        text_pos_ids_batch = []
        text_ent_ids_batch = []
        title_word_ids_batch = []
        for idx in range(1000):
            text_sents_ids, text_sents_pos, text_sents_ent, title_word_ids = \
                self.dev_dt[idx].get_all()
            text_word_ids_batch.append(text_sents_ids)
            text_pos_ids_batch.append(text_sents_pos)
            text_ent_ids_batch.append(text_sents_ent)
            title_word_ids_batch.append(title_word_ids)
        return text_word_ids_batch, text_pos_ids_batch, text_ent_ids_batch, title_word_ids_batch

    def get_test_all(self):
        """
        获取dev文件作为测试集
        :return:
        """
        text_word_ids_batch = []
        text_pos_ids_batch = []
        text_ent_ids_batch = []
        for idx in range(1000):
            text_sents_ids, text_sents_pos, text_sents_ent, _ = self.test_dt[idx].get_all()
            text_word_ids_batch.append(text_sents_ids)
            text_pos_ids_batch.append(text_sents_pos)
            text_ent_ids_batch.append(text_sents_ent)
        return text_word_ids_batch, text_pos_ids_batch, text_ent_ids_batch
