# seedcup2018

[mainRefer](https://github.com/wabyking/TextClassificationBenchmark)

code about [种子杯](http://rank.dian.org.cn/static/index.html)

暂时没有公开的训练数据

## 记号

`typedef pandas.DataFrame df`

## env prerequisite

+ 不限平台
+ python 3.*(以下没有特别说明,均为2018-10最新)

  + pandas
    + numpy
  + sklearn
  + pytorch(pytorch-cpu allowed)
    + torchtext
  + gensim

## data view

![data.png](visualization.png)

## directory structure

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
    ├── raw\                data the official provide
    ├── train.py            support model train
    ├── util.py             support data process
    └── ...                 other files

## how to run

generate processed data in `data/` (need data in `raw/`)
>`py datahelper.py` 

suppose you want to save model in  `abc` dir

>`py main.py abc`

finally it will generate predicate `txt` for `raw\test_b.txt`

### load model and train

in `config.py`

change para `LAST_EPOCH`  ; and `LOADMODEL` to where the model saved

>`py main.py abc`

### modify answer manual

use ipython to run `main.py`

after run
>`ans = util.get_pred_list(model, test_iter, use_pandas=True)`

you will get a `df` ans

## model parameters

name | usage
--|--
MAX_EPOCH | num of train epoch
BATCH_SIZE=640  | usage 1G GPU when set to 64
MAX_SEQ_LEN=200 | fixed and max length of word
NUM_LAYER=2 | num of recurrent layers, stacking two LSTM together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
DROPOUT | dropout probability of Dropout layer
wei_criterion | used to calculate total loss

others refer to `config.py`

## interface

func    |usage
--|--
get_pred_list | get predict for  buck_iter, return `2dlist`
get_pred_pd   | get predict for  buck_iter, return `pd`
creterion_val | input `2dlist` or `df`, return the score in validset