# Intro

This is the code about [种子杯](http://rank.dian.org.cn/static/index.html), a chinese text multi-classification contest

It shows how to build word embedding by **gensim** and use **torchtext** to process the data, and, finally training a **BLSTM** model in **pytorch**.

## project feature

+ around 300 lines
+ only has basic function
+ shared for educational purpose
+ scored 86.42 in semi-final, ranked in the 6th echelone

# refer

1. [mainRefer](https://github.com/wabyking/TextClassificationBenchmark)
2. [Attention-Based BLSTM](http://www.aclweb.org/anthology/P16-2034)

# denote

`typedef pandas.DataFrame df`

# env prerequisite

+ OS: run on windows and linux
+ python 3.*(below are newest in 2018-10)
  + pandas
    + numpy
  + sklearn
  + pytorch(pytorch-cpu allowed)
    + torchtext
  + gensim

# data view

[dataset location](https://www.kaggle.com/lyf9828/seedcup2018/home)

item_id | title_characters        |title_words     |description_characters|  description_words |      cate1_id        |cate2_id|        cate3_id|
--|--|--|--|--|--|--|--
a38b804b6eb25c6a39eef30e54060ce1|c51,c38,c48,c45,c10,c7,c288,c18,c15,c7,c255,c305,c18,c56,c762,c549,c1051,c18,c1051,c147,c955,c259,c18|w27,w12,w22,w215,w11,w875,w1242,w14391,w4018,w5656|c32,c540,c101,c275,c613,c61,c92,c54,c467,c354,c361,c61,c154,c183,c247,c71,c398,c21,c31,c2,c32,c23,c135,c229,c1175,c61,c76,c23,c135,c982,c71,c2,c1175,c633,c195,c61,c62,c197,c61,c14,c1163,c166,c31|w8,w295,w2132,w13,w86,w1830,w3009,w13,w167,w395,w1499,w4,w7,w8,w87,w3584,w13,w93,w87,w2014,w3843,w13,w111,w13,w14,w2867,w7|2|13|13

<!-- ![data.png](visualization.png) -->
<img align="right" src= "visualization.png">
*one catej_id corresponding only one catei_id, for j>i*

<<<<<<< HEAD
# directory structure
=======
## directory structure
>>>>>>> d025d735022d4978c24d44e46af2c3f3fb42ccb7

    .
    ├── config.py           
    ├── data                
    │   ├── test_w.tsv      
    │   ├── train_w.tsv     
    │   ├── val_w.tsv       
    │   └── w300.txt        gensim model saved
    ├── datahelper.py       preprocess data
    ├── main.py             
    ├── model\              model usaged
    ├── doc\                context explain and report
    ├── raw\                here put data provided
    ├── train.py            support model train
    ├── util.py             support data process
    └── ...                 other files

# how to run

generate processed data in `data/` (need data in `raw/`)
>`py datahelper.py` 

suppose you want to save model in  `abc` dir

>`py main.py abc`

finally it will generate predicate `txt` for `raw\test_b.txt`

## load model and train

in `config.py`

change para `LAST_EPOCH`  ; and `LOADMODEL` to where the model saved

>`py main.py abc`

## modify answer manual

use ipython to run `main.py`

after run
>`ans = util.get_pred_list(model, test_iter, use_pandas=True)`

you will get a `df` ans

# model parameters

name | usage
--|--
MAX_EPOCH | num of train epoch
MAX_SEQ_LEN=200 | fixed and max length of word
NUM_LAYER=2 | num of recurrent layers, stacking two LSTM together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
DROPOUT | dropout probability of Dropout layer
wei_criterion | used to calculate total loss

others refer to `config.py`

# interface

func    |usage
--|--
get_pred_list | get predict for  buck_iter, return `2dlist`
get_pred_pd   | get predict for  buck_iter, return `pd`
creterion_val | input `2dlist` or `df`, return the score in validset
