# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: Pre-process the data provided by ByteCup
"""
import json
from config import *
from utils.file_util import *
from utils.text_util import get_sent_words_syns
from models.doc_struct import Document


def build_voc():
    """
    构建word2ids即可，低频词过滤，因为解码需要得到特殊的，所以暂时不做低频词过滤
    对每个篇章的数据直接构建成doc对象
    :return:
    """
    # 数据读取
    data_list = [TRAIN_RAW, DEV_RAW]
    # 词汇库构建
    word2ids = dict()
    word_freq = dict()
    word2ids[UNK] = 0
    word2ids[PAD] = 1
    word_ids = 2
    count_line = 0
    for dir_name in data_list:
        for file_name in os.listdir(dir_name):
            tmp_documents_list = []
            if file_name.endswith(".txt"):  # 1, 4, 5, 6, 有问题
                print("当前处理文件名：", file_name)
                file_path = os.path.join(dir_name, file_name)
                with open(file_path, "r") as f:
                    for line in f:
                        tmp_document = Document()
                        count_line += 1
                        try:
                            tmp_line_ids = []
                            # 对文档内容的token统计
                            doc_tokens = get_sent_words_syns(json.loads(line.strip())["content"])
                            for tok in doc_tokens:
                                if tok in word2ids.keys():
                                    word_freq[tok] += 1
                                    tmp_line_ids.append(word2ids[tok])
                                else:
                                    word_freq[tok] = 1
                                    word2ids[tok] = word_ids
                                    word_ids += 1
                                    tmp_line_ids.append(word2ids[tok])
                            tmp_document.text_word_ids = tmp_line_ids
                            tmp_document.text_tokens = doc_tokens
                        except Exception:
                            print(line)
                            print("line_number: ", count_line)
                            input("content error!")
                            exit()
                        if dir_name != DEV_RAW:
                            try:
                                tmp_title_ids = []
                                # 对标题的token统计
                                title_tokens = get_sent_words_syns(json.loads(line.strip())["title"])
                                for tok in title_tokens:
                                    if tok in word2ids.keys():
                                        word_freq[tok] += 1
                                        tmp_title_ids.append(word2ids[tok])
                                    else:
                                        word_freq[tok] = 1
                                        word2ids[tok] = word_ids
                                        word_ids += 1
                                        tmp_title_ids.append(word2ids[tok])
                                tmp_document.title_word_ids = tmp_title_ids
                                tmp_document.title_tokens = title_tokens
                            except Exception:
                                print(line)
                                print("line_number: ", count_line)
                                print("title error!")
                                input(line)
                                exit()
                        tmp_documents_list.append(tmp_document)
                # 累积存储
                save_data((word2ids, word_ids), WORD2IDS)
                save_data((word_freq, word_ids), WORD_FREQ)
                save_path = TRAIN_documents if dir_name != DEV_RAW else DEV_documents
                save_data(tmp_documents_list, os.path.join(save_path, file_name.replace(".txt", ".pkl")))
    print("successfully build vocabulary!")


def build_train_dev():
    """
    创建针对train和dev的ids数据集，对每个篇章的数据直接构建成doc对象
    :return:
    """
    ...


if __name__ == "__main__":
    build_voc()
