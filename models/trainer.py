# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: Deal with training and validation
"""
import torch
from rouge import FileRouge
from config import *
import torch.nn as nn
import torch.optim as optim
from utils.file_util import *
from models.general_model import Lemon_Model
from utils.data_iterator import data_iterator


class Trainer:
    def __init__(self):
        self.data_iterator_ = data_iterator()
        self.train_dt = self.data_iterator_.next_train_batch()
        self.dev_dt = self.data_iterator_.get_dev_all()
        self.test_dt = self.data_iterator_.get_test_all()
        self.model = Lemon_Model()
        self.ids2word = load_data(IDS2WORD)
        self.log_file = LOG_FILE_PATH

    def do_train(self):
        """
        训练指定轮数之后评测出结果提交平台
        :return:
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=L2_penalty)
        optimizer.zero_grad()
        loss_distribution = []
        batch_count = 0
        for epoch in range(EPOCH_ALL):
            try:
                while True:
                    batch_count += 1
                    # all_ids前三个数据项：(batch_size, sent_num, sent_len)
                    all_ids = next(self.train_dt)
                    # 批量数据解码和对应的一批labels直接计算loss并学习
                    batch_out = self.model(all_ids)
                    batch_labels = all_ids[3]
                    print(batch_labels.size())
                    input(batch_out.size())
                    loss_ = criterion(batch_out, batch_labels)
                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # 打印信息
                    loss_of_batch = loss_.data.item() / BATCH_SIZE
                    loss_str = "the train loss_tran is: " + str(loss_of_batch)
                    print_(loss_str, self.log_file)
                    # loss分布分析
                    loss_distribution.append(loss_of_batch)
                    if batch_count > 0 and batch_count % SKIP_STEPS == 0:
                        # 计算准确率
                        text_word_ids_batch, text_pos_ids_batch, text_ent_ids_batch, title_word_ids_batch = self.dev_dt
                        title_list = self.model.title_generate((text_word_ids_batch, text_pos_ids_batch, text_ent_ids_batch))
                        self.get_pre(title_list, title_word_ids_batch)
            except TimeoutError:
                input("一次迭代结束，打乱数据新一轮迭代")
                self.data_iterator_.new_epoch()
        # 最终评测
        self.do_eval()
        # loss 存储
        save_data(loss_distribution, TRAIN_loss_path)
        # model 存储
        torch.save(self.model.state_dict(), Pretrained_model)

    def do_eval(self):
        """
        对test集合的标题生成检测，挑出最好模型之后做
        :return:
        """
        title_list = self.model.title_generate(self.test_dt)
        self.write_dev_test(title_list, TEST_OUTPUT_PATH)
        print("请将生成的标题提交至评测平台...")

    def get_pre(self, title_list_out, title_list_gold):
        """
        输入模型预测的结果和标准结果，计算准确率
        :param title_list_out: (batch_size, title_length)
        :param title_list_gold: (batch_size, title_length)
        :return:
        """
		# 将得到的结果转成一行一行的形式
        self.write_dev_test(title_list_out, DEV_OUTPUT_PATH)
        self.write_dev_test(title_list_gold, DEV_GOLD_PATH)
        # 调用rouge对两个文件的数据进行
        files_rouge = FilesRouge(DEV_OUTPUT_PATH, DEV_GOLD_PATH)
        scores = files_rouge.get_scores(avg=True)
        print("当前在开发集上的rouge平均准确率：", scores[0]["rouge-1"]["p"])

    def write_dev_test(self, title_list, dist_path):
        """
        将生成的数据存储到指定文件
        """
        for title_ids in title_list:
            line_ = ""
            for word_id in title_ids:
                line_ += self.ids2word[word_id]
            write_append(line, dist_path)
