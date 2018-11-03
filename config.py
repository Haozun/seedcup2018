# common libs
import torch as tr
import torch.nn as nn
import numpy as np
import sys
from datetime import datetime

# %% machine independent
EPOCHS = 20
USE_GPU = 1
NUM_GPU = 2
DEVICE = tr.device("cuda" if tr.cuda.is_available() and USE_GPU else "cpu")
THREAD = 12
tr.set_num_threads(THREAD)

CATE1_NUM = 20
CATE2_NUM = 135
CATE3_NUM = 265

NUM_LAYERS = 2  # Stack

MAX_SEQ_LEN = 200  # default torchtext field
GRAD_CLIP = 0.1  # default, Gradient explosion


def wei_criterion(x):
    return 0.1 * x[0] + 0.3 * x[1] + 0.6 * x[2]


DROPOUT = 0.2  # default 0.2
HIDDEN_DIM = 256  # default 128

BATCH_SIZE = 64
tr.manual_seed(1)
tr.set_num_threads(THREAD)
# %% file location
try:
    prodirectory = sys.argv[1]
    if prodirectory == '-f':
        raise IndexError
except IndexError:
    prodirectory = str(datetime.now())[11:16]
    print("workspace was set to " + prodirectory)

EMBEDDING_DIM = 300
LOADMODEL=None
LAST_EPOCH=0
W2VFILE = "data/w300.txt"

TRAINFILE = "data/train_w.tsv"
VALFILE = "data/val_w.tsv"
TESTFILE = "data/test_w.tsv"

OTRAIN = "raw/train_b.txt"
OTEST = "raw/test_b.txt"
OVAL = "raw/valid_b.txt"
OTEST_A = "raw/test_a.txt"
OTRAIN_A = "raw/train_a.txt"
OVAL_A = "raw/valid_a.txt"
