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
DEV_documents = "data/data_documents/DEV/bytecup.corpus.train.7.pkl"
TEST_documents = "data/data_documents/DEV/bytecup.corpus.validation_set.pkl"
WORD2IDS = "data/word2ids.pkl"
IDS2WORD = ""
SENT_NUM_FREQ = "data/data_documents/sent_num_freq.pkl"
SENT_LEN_FREQ = "data/data_documents/sent_len_freq.pkl"
TRAIN_loss_path = "data/train_loss_distribution.pkl"
DEV_OUTPUT_PATH = "data/dev_out_titles.tsv"
DEV_GOLD_PATH = "data/dev_gold_titles.tsv"
TEST_OUTPUT_PATH = "data/test_out_titles.tsv"

# experimental
SET_VERSION = "1"
TRAIN_SET_NUM = 8

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
END = "<END>"
UNK_ID = 0
PAD_ID = 1
END_ID = 2

# PAD 参数
MAX_TITLE_LENGTH = 20
MAX_SENT_LENGTH = 20  # 根据统计得到
MAX_SENT_NUMBER = 20  # 根据统计得到


# 神经网络参数
LR = 0.003
L2_penalty = 1e-5
BATCH_SIZE = 130
SKIP_STEPS = 100
EPOCH_ALL = 7  # 整个数据集迭代3, 5, 7次
RANDOM_SEED = 2
HIDDEN_SIZE = 1024

DECODE_OUT_SIZE = 200

# Model2Save
Pretrained_model = "data/pre_trained_model.pth"
LOD_DIR = "data/log_file/"
LOG_FILE_PATH = "data/log_file/version_" + SET_VERSION + ".log"

