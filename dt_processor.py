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
from stanfordcorenlp import StanfordCoreNLP

path_to_jar = 'stanford-corenlp-full-2018-02-27'
nlp = StanfordCoreNLP(path_to_jar)

def build_voc():
    """
    将词分成tokens进行存储，统计好词频信息，word2ids词典等
    :return:
    """
    # 数据读取
    data_list = [TRAIN_RAW, DEV_RAW]
    word_freq = dict()
    title_word_set = set()
    for dir_name in data_list:
        for file_name in os.listdir(dir_name):
            tmp_documents_list = []
            if file_name.endswith(".txt"):  # 1, 4, 5, 6, 有问题
                print("当前处理文件名：", file_name)
                file_path = os.path.join(dir_name, file_name)
                with open(file_path, "r") as f:
                    for line in f:
                        tmp_document = Document()
                        doc_tokens = get_sent_words_syns(json.loads(line.strip())["content"])
                        for tok in doc_tokens:
                            word_freq[tok] = word_freq[tok] + 1 if tok in word_freq.keys() else 1
                        tmp_document.text_tokens = doc_tokens
                        # 训练集中标题部分词汇分析
                        if dir_name != DEV_RAW:
                            # 对标题的token统计
                            title_tokens = get_sent_words_syns(json.loads(line.strip())["title"])
                            for tok in title_tokens:
                                title_word_set.add(tok)
                                word_freq[tok] = word_freq[tok] + 1 if tok in word_freq.keys() else 1
                            tmp_document.title_tokens = title_tokens
                        tmp_documents_list.append(tmp_document)
            # 累积存储
            save_path = TRAIN_documents if dir_name != DEV_RAW else DEV_documents
            save_data(tmp_documents_list, os.path.join(save_path, file_name.replace(".txt", ".pkl")))
    # 下面对正文部分的低频词进行过滤
    word2ids_ = low_freq_filter(word_freq, title_word_set)
    save_data(word2ids_, WORD2IDS)


def low_freq_filter(word_freq, title_word_set):
    """
    根据统计词频和是否在标题生成最佳的word2ids信息
    :return:
    """
    # 词汇库构建
    word2ids_ = dict()
    word2ids_[UNK] = UNK_ID
    word2ids_[PAD] = PAD_ID
    word2ids_[END] = END_ID
    word_ids = 3
    for word in word_freq.keys():
        if ((word_freq[word]) > 1 or (word in title_word_set)) and (word not in word2ids_.keys()):
            word2ids_[word] = word_ids
            word_ids += 1
    return word2ids_


def sent_seg(word2ids_=None):
    """
        根据封装好的document对象，将数据中content以及对应的pos，ent，token进行分局处理
        转换完的document对象的list存储在TRAIN_documents_tran下

        加载词频统计情况，对低频词的过滤，注意这里只过滤来自文本内的低频词，标题的低频词需要保留
    :return:
    """
    # statistic 统计每句话里面的句长信息， 统计每个文章里面的句子数信息
    sent_length_freq = dict()
    sent_num_freq = dict()
    title_length_freq = dict()
    for doc_dir in [TRAIN_documents, DEV_documents]:
        for file_name in os.listdir(doc_dir):
            if file_name.endswith("pkl"):
                print("The temporary seg documents: ", file_name)
                file_path = os.path.join(doc_dir, file_name)
                documents = load_data(file_path)
                document_list = []
                for document in documents:
                    tmp_sent_ids_list = []
                    tmp_sent_words_list = []
                    # 对每句话转id
                    tmp_sent_ids = []
                    tmp_sent_words = []
                    for token in document.text_tokens:
                        token_id = word2ids_[token] if token in word2ids_.keys() else UNK_ID
                        tmp_sent_ids.append(token_id)
                        tmp_sent_words.append(token)
                        if token == ".":
                            tmp_sent_ids_list.append(tmp_sent_ids)
                            tmp_sent_len = len(tmp_sent_ids)
                            # 统计句长
                            sent_length_freq[tmp_sent_len] = sent_length_freq[tmp_sent_len] + 1 \
                                if tmp_sent_len in sent_length_freq.keys() else 1
                            tmp_sent_words_list.append(tmp_sent_words)
                            tmp_sent_ids = []
                            tmp_sent_words = []
                    document.text_sents_ids = tmp_sent_ids_list
                    document.text_sents_tokens = tmp_sent_words_list

                    # 对标题转id
                    tmp_title_ids = []
                    for token in document.title_tokens:
                        token_id = word2ids_[token] if token in word2ids_.keys() else UNK_ID
                        tmp_title_ids.append(token_id)
                    tmp_title_ids.append(END_ID)
                    document.title_word_ids = tmp_title_ids
                    document_list.append(document)
                    # 统计
                    tmp_sent_num = len(tmp_sent_ids_list)
                    sent_num_freq[tmp_sent_num] = sent_num_freq[tmp_sent_num] + 1 \
                        if tmp_sent_num in sent_num_freq.keys() else 1
                    tmp_title_len = len(tmp_title_ids)
                    title_length_freq[tmp_title_len] = title_length_freq[tmp_title_len] + 1 \
                        if tmp_title_len in title_length_freq.keys() else 1
                save_data(document_list, file_path)  # 覆盖源文件
    save_data(sent_length_freq, SENT_LEN_FREQ)
    save_data(sent_num_freq, SENT_NUM_FREQ)


def build_full_documents():
    """
    对所有文档的按照句子列表存储的形式获取对应 ENT and POS 等信息
    完成对句子的padding以及对文章句子的padding操作
    :return:
    """
    pos2ids = dict()
    ent2ids = dict()
    # document 最终
    document_list = []
    for doc_dir in [TRAIN_documents, DEV_documents]:
        for file_name in os.listdir(doc_dir):
            if file_name.endswith(".pkl"):
                file_path = os.path.join(doc_dir, file_name)
                documents = load_data(file_path)
                for document in documents:
                    # 对内容部分的pos and ent生成
                    tmp_pos_ids_list = []
                    tmp_ent_ids_list = []
                    tmp_token_ids_list = []
                    tmp_token_list = []
                    for sent_tokens, sent_ids in zip(document.text_sents_tokens, document.text_sents_ids):
                        tmp_pos_ids = []
                        tmp_ent_ids = []
                        ent_ids = pos_ids = 0
                        pos_tags, ent_tags = get_pos_ent(sent_tokens)
                        for pos_, ent_ in zip(pos_tags, ent_tags):
                            pos_id = pos2ids[pos_] if pos_ in pos2ids.keys() else pos_ids
                            if pos_id == pos_ids:
                                pos2ids[pos_] = pos_ids
                                pos_ids += 1
                            ent_id = ent2ids[ent_] if ent_ in ent2ids.keys() else ent_ids
                            if ent_id == ent_ids:
                                ent2ids[pos_] = ent_ids
                                ent_ids += 1
                            tmp_pos_ids.append(pos_id)
                            tmp_ent_ids.append(ent_id)

                        # 对4类数据的padding操作
                        while len(sent_ids) < MAX_SENT_LENGTH:
                            sent_tokens.appent(PAD)
                            sent_ids.appent(PAD_ID)
                            tmp_pos_ids.append(PAD_ID)
                            tmp_ent_ids.append(PAD_ID)
                        tmp_pos_ids_list.append(tmp_pos_ids)
                        tmp_ent_ids_list.append(tmp_ent_ids)
                        tmp_token_ids_list.append(sent_ids)
                        tmp_token_list.append(sent_tokens)
                    # 对篇章句子数的padding
                    while len(tmp_token_list) < MAX_SENT_NUMBER:
                        pad_general = [0 for _ in range(MAX_SENT_LENGTH)]
                        tmp_pos_ids_list.append(pad_general)
                        tmp_ent_ids_list.append(pad_general)
                        tmp_token_ids_list.append(pad_general)
                        tmp_token_list.append(pad_general)
                    # 对标题长度的padding
                    tmp_title_ids = document.title_word_ids
                    while len(tmp_title_ids) < MAX_TITLE_LENGTH:
                        tmp_title_ids.append(PAD_ID)
                    # 集体更新
                    document.text_sents_pos = tmp_pos_ids_list
                    document.text_sents_ent = tmp_ent_ids_list
                    document.text_sents_ids = tmp_token_ids_list
                    document.text_sents_tokens = tmp_token_list
                    document.title_word_ids = tmp_title_ids
                    document_list.append(document)
                save_data(document_list, file_path)  # 覆盖存储


def test_seg():
    for file_name in os.listdir(TRAIN_documents):
        if file_name.endswith(".pkl"):
            documents = load_data(os.path.join(TRAIN_documents, file_name))
            for document in documents:
                print(document.text_sents_tokens)
                input()


def get_pos_ent(sent_tokens):
    """
    根据tokens获取这句话的pos和ent信息
    :param sent_tokens:
    :return:
    """
    sent_str = " ".join(sent_tokens)
    pos_tags = [pair[1] for pair in nlp.pos_tag(sent_str)]
    ent_tags = [pair[1] for pair in nlp.ner(sent_str)]
    return pos_tags, ent_tags


if __name__ == "__main__":
    build_voc()
    # word2ids = load_data(WORD2IDS)
    # sent_seg(word2ids_=word2ids)
    # 创建其他信息，统计词频之后在config中确定padding长度并进行下面的操作
    # build_full_documents()
    # test_seg()
