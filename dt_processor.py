# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: Pre-process the data provided by ByteCup
"""
import json
from config import *
from utils.file_util import *
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(STANFORD_JAR)


def build_voc():
    """
    构建word2ids即可，低频词过滤，因为解码需要得到特殊的，所以暂时不做低频词过滤
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
    for dir_name in data_list:
        for file_name in os.listdir(dir_name):
            tmp_data_ids = []
            tmp_titles_ids = []
            if file_name.endswith(".txt"):
                file_path = os.path.join(dir_name, file_name)
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            tmp_line_ids = []
                            # 对文档内容的token统计
                            doc_tokens = nlp.word_tokenize(json.loads(line.strip())["content"])
                            for tok in doc_tokens:
                                if tok in word2ids.keys():
                                    word_freq[tok] += 1
                                    tmp_line_ids.append(word2ids[tok])
                                else:
                                    word_freq[tok] = 1
                                    word2ids[tok] = word_ids
                                    word_ids += 1
                                    tmp_line_ids.append(word2ids[tok])
                            tmp_data_ids.append(tmp_line_ids)
                        except Exception:
                            print("content error!")
                            input(line)
                        if dir_name != DEV_RAW:
                            # dev 中不提供title，所以数据处理成ids的过程不涉及对dev集合的数据处理
                            try:
                                tmp_title_ids = []
                                # 对标题的token统计
                                title_tokens = nlp.word_tokenize(json.loads(line.strip())["title"])
                                for tok in title_tokens:
                                    if tok in word2ids.keys():
                                        word_freq[tok] += 1
                                        tmp_title_ids.append(word2ids[tok])
                                    else:
                                        word_freq[tok] = 1
                                        word2ids[tok] = word_ids
                                        word_ids += 1
                                        tmp_title_ids.append(word2ids[tok])
                                tmp_titles_ids.append(tmp_title_ids)
                            except Exception:
                                print("title error!")
                                input(line)
                # 累积存储
                save_data((word2ids, word_ids), WORD2IDS)
                save_data((word_freq, word_ids), WORD_FREQ)

                save_data(tmp_data_ids, dir_name + file_name.replace(".txt", ".pkl"))
                save_data(tmp_titles_ids, dir_name + file_name.replace(".txt", ".labels.pkl"))
    print("successfully build vocabulary!")


def build_train_dev():
    """
    创建针对train和dev的ids数据集
    :return:
    """
    ...


if __name__ == "__main__":
    build_voc()
