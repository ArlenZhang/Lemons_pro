# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description:
"""
from utils.file_util import *
from config import *

if __name__ == "__main__":
    train_dt = load_data(TRAIN_PP)
    dev_dt = load_data(DEV_PP)
    test_dt = load_data(TEST_PP)

    # trainer train

