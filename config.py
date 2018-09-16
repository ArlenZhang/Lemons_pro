# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description:
"""
# external package
STANFORD_JAR = r'stanford-corenlp-full-2018-02-27'

# data path
TRAIN_RAW = "data/data_raw/TRAINING"
DEV_RAW = "data/data_raw/DEV"

TRAIN_documents = "data/data_documents/TRAINING"
DEV_documents = "data/data_documents/DEV"

TRAIN_loss_path = "data/train_loss_distribution.pkl"

# 词参数
EMBED_SIZE = 512
WORD_LEN = 30000
POS_EMB_SIZE = 10
POS_LEN = 30
ENT_EMB_SIZE = 10
ENT_LEN = 20
DIST_EMB_SIZE = 10
DIST_LEN = 30
UNK = "<UNK>"
PAD = "<PAD>"
WORD2IDS = "data/word2ids.pkl"
WORD_FREQ = "data/word2freq.pkl"

# 神经网络参数
LR = 0.003
L2_penalty = 1e-5
BATCH_SIZE = 130
EPOCH_ALL = 7  # 整个数据集迭代3, 5, 7次
RANDOM_SEED = 2
HIDDEN_SIZE = 1024

DECODE_OUT_SIZE = 200
MAX_TITLE_LENGTH = 20

