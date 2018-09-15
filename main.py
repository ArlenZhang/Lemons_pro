# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description:
"""
from utils.file_util import *
from config import *
from dt_processor import build_voc, build_train_dev
from sys import argv
from models.trainer import Trainer


def prepare_data():
    """
    创建词库、构建ids训练数据等
    :return:
    """
    build_voc()
    build_train_dev()


if __name__ == "__main__":
    if len(argv) >= 2:
        test_desc = argv[1]
    else:
        test_desc = "no message."
    safe_mkdirs([TRAIN_documents, DEV_documents])
    prepare_data()

    # 加载数据
    train_dt = load_data(TRAIN_documents)
    dev_dt = load_data(DEV_documents)

    # trainer train
    trainer = Trainer(train_dt, dev_dt)
    trainer.do_train()
