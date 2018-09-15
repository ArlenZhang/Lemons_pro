# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description:
"""
from config import *
from stanfordcorenlp import StanfordCoreNLP

nlp_ = StanfordCoreNLP(STANFORD_JAR)

title_plain_text = "I am ok today."
title_tokens = nlp_.word_tokenize(title_plain_text)
print(title_tokens)
