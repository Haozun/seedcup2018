from config import *
import os
import numpy as np
import pandas as pd
from torchtext import data, vocab
from sklearn.metrics import f1_score

multi_classes = []
valid_set = pd.read_csv(VALFILE, sep='\t')
test_set = pd.read_csv(OTEST, sep='\t')


def init_workspace():
    if not os.path.exists(prodirectory):
        print("directory at " + prodirectory)
        os.makedirs(prodirectory)
    else:
        print("warning: directory already exists")

    global multi_classes
    multi_classes = [data.LabelField() for _ in range(3)]
    word_field = data.Field(tokenize=lambda x: x.split(','),
                            include_lengths=True, batch_first=True, fix_length=MAX_SEQ_LEN)

    print("load torch data ")
    class_fields = [('w', word_field),
                    ('cate1_id', multi_classes[0]), ('cate2_id', multi_classes[1]), ('cate3_id', multi_classes[2])]
    train = data.TabularDataset(TRAINFILE, 'tsv', skip_header=True, fields=class_fields)
    valid = data.TabularDataset(VALFILE, 'tsv', skip_header=True, fields=class_fields)
    test = data.TabularDataset(TESTFILE, 'tsv', skip_header=True, fields=[('w', word_field)])
    # discretization
    word_field.build_vocab(train, valid, test)

    for cls in multi_classes:
        cls.build_vocab(train, valid)

    trainiter = data.BucketIterator(train, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.w), shuffle=True)
    valiter = data.BucketIterator(valid, batch_size=BATCH_SIZE, shuffle=False)
    testiter = data.BucketIterator(test, batch_size=BATCH_SIZE, shuffle=False)

    vectors = vocab.Vectors(W2VFILE)
    print("Word2vec model Loaded")
    word_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

    return word_field.vocab.vectors, trainiter, valiter, testiter


def get_pred_list(model, buck_iter):
    # use_pandas: whether return preds list or a pandas file build from %template_file%
    model.eval()
    model_preds = [[], [], []]
    for _, batch in enumerate(buck_iter):
        text = batch.w[0]
        model_preds_batch = model(text.to(DEVICE))
        for i in range(3):
            model_preds[i] += model_preds_batch[i].argmax(1).tolist()

    return model_preds


def get_pred_pd(model, buck_iter, template_file=OTEST):
    pred_list = get_pred_list(model, buck_iter)
    discretize(pred_list, reverse=True)

    ans = pd.read_csv(template_file, sep='\t')
    if template_file is OTEST:
        ans.drop(columns=["title_characters", "title_words", "description_characters", "description_words"],
                 inplace=True)

    for i in range(3):
        for j in range(len(ans)):
            ans.at[j, "cate%d_id" % (i + 1)] = pred_list[i][j]
        ans["cate%d_id" % (i + 1)] = ans["cate%d_id" % (i + 1)].astype(int)
    return ans


def creterion_val(preds_list, df=valid_set):  # implicit: if preds_list is list, it will be changed
    if type(preds_list) is pd.core.frame.DataFrame:
        f = [f1_score(df["cate%d_id" % (i + 1)], preds_list["cate%d_id" % (i + 1)], average="macro") for i in
             range(3)]
        fa = [f1_score(df["cate%d_id" % (i + 1)], preds_list["cate%d_id" % (i + 1)], average="micro") for i in
              range(3)]
    else:
        discretize(preds_list, reverse=True)
        f = [f1_score(df["cate%d_id" % (i + 1)], preds_list[i], average="macro") for i
             in range(3)]

        fa = [f1_score(df["cate%d_id" % (i + 1)], preds_list[i], average="micro") for i
              in range(3)]
    
    fw = np.dot([0.1, 0.3, 0.6], f)
    return f + fa + [fw]


def log_and_print(eval_res):
    if type(eval_res[0]) is tr.Tensor:
        file = open(prodirectory + "/loss", 'a')
        eval_str = "loss %.4f %.4f %.4f %.4f \n" % tuple(eval_res)
        # print(eval_res)
        file.write(eval_str)
    else:
        file = open(prodirectory + "/log", 'a')
        eval_res = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f" % tuple(eval_res) + '\n'
        print(eval_res)
        file.write(eval_res)


def discretize(input_list, reverse=False):  # inplace
    if reverse is False:
        ordered_map_lists = list(map(lambda x: x.vocab.stoi, multi_classes))
        for i in range(3):
            input_list[i] = list(map(lambda e: ordered_map_lists[i][str(e)], input_list[i]))
    else:
        reverse_map_list = list(map(lambda x: x.vocab.itos, multi_classes))
        for i in range(3):
            input_list[i] = list(map(lambda e: int(reverse_map_list[i][e]), input_list[i]))
