# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: Deal with training and validation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from config import *
from utils.file_util import *
from models.general_model import Lemon_Model


class Trainer:
    def __init__(self, train_dt=None, dev_dt=None):
        self.train_dt = train_dt
        self.dev_dt = dev_dt
        self.model = Lemon_Model()
        self.log_file = ...

    def do_train(self):
        """
        训练指定轮数之后评测出结果提交平台
        :return:
        """
        random.seed(RANDOM_SEED)  # 保证别人再次跑的时候有相同的效果
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=L2_penalty)
        optimizer.zero_grad()
        batch_scores = None
        batch_labels = None
        loss_distribution = []
        iter_count = 0
        batch_count = 0
        loss_ = 0.
        for epoch in range(EPOCH_ALL):
            random.shuffle(self.train_dt)
            for document in self.train_dt:
                iter_count += 1
                doc_score = self.model(document)
                batch_scores = doc_score if batch_scores is None else torch.cat((batch_scores, doc_score), 0)
                batch_labels = document.title_word_ids if batch_labels is None else \
                    torch.cat((batch_labels, document.title_word_ids), 0)
                loss_ += criterion(batch_scores, torch.Tensor(batch_labels).long())
                if iter_count % BATCH_SIZE == 0 and iter_count > 0:
                    batch_count += 1
                    # compute the grads of the loss function according to this batch of data
                    optimizer.zero_grad()
                    # Margin loss
                    loss_.backward(retain_graph=True)  # retain_graph=True
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_of_batch = loss_.data.item() / BATCH_SIZE
                    loss_str = "the train loss_tran is: " + str(loss_of_batch)
                    print_(loss_str, self.log_file)
                    # loss分布分析
                    loss_distribution.append(loss_of_batch)
                    loss_ = 0.
        self.do_eval()
        self.report()
        # loss 存储
        save_data(loss_distribution, TRAIN_loss_path)

    def do_eval(self):
        random.shuffle(self.dev_dt)
        for doc_ids in self.dev_dt:
            doc_score = self.model(doc_ids)
            title_predicted = self.ids2words(doc_score)
            write_append(title_predicted, ..., type_="over")
        print("请将生成的标题提交至评测平台...")

    @staticmethod
    def ids2words(doc_score):
        words_of_title = ""
        for ids in doc_score:
            word = ...
            words_of_title += word + " "
        return words_of_title

    def report(self):
        ...
