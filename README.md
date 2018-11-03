# seedcup2018

[主页](https://uniqueai.me/seedcup2018/)

[主要参考](https://github.com/wabyking/TextClassificationBenchmark)

关于[种子杯复赛](http://rank.dian.org.cn/static/index.html)的代码, 总行数200行左右

暂时没有训练数据

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

## 数据概览

![data.png](visualization.png)

## 目录结构

    ├── answerlastlong.txt  预测后的文件
    ├── config.py           主要配置文件
    ├── data                处理后的数据;运行datahelper后生成的文件
    │   ├── test_w.tsv      测试集
    │   ├── train_w.tsv     训练集
    │   ├── val_w.tsv       验证集
    │   └── w300.txt        词向量模型
    ├── datahelper.py       预处理数据
    ├── main.py             主程序
    ├── model\              所使用的模型
    ├── doc\                比赛说明文档及报告
    ├── raw\                比赛方提供的数据
    ├── train.py            提供训练代码支持
    ├── util.py             提供数据处理支持
    └── ...                 其他文件

## 运行

如果缺失data数据,用 `py datahelper.py` 生成(需要`raw/`文件夹的原始数据)

运行时首先保证你当前的目录结构与上面一致

假设你要保存模型在 `abc` 文件夹

`py main.py abc`

输入`data\`里的数据,将会生成对`data\test_w.tsv` 的预测结果

### 载入模型

修改 `config.py` 里面的 `LAST_EPOCH` 参数 ; 并把 `LOADMODEL` 参数设置为 模型路径

 `py main.py abc`

### 手动修改答案

使用 ipython 逐条运行 `main.py` (其实也就5行)

`ans = util.get_pred_list(model, test_iter, use_pandas=True)`

此处返回值为 `pd.DataFrame` 格式, 可以手动修改

## 参数

name | usage
--|--
MAX_EPOCH | 训练轮数
BATCH_SIZE=640  | 设置为64时,占用GPU 1G; 增大可加速训练, 但是精度下降 ; 建议设置大点
MAX_SEQ_LEN=200 | 最大的`t_w+d_w`的长度
NUM_LAYER=2 | num of recurrent layers, stacking two LSTM together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
DROPOUT | dropout probability of Dropout layer
wei_criterion | used to calculate total loss

其他参数见`config.py`

## 接口

func    |usage
--|--
get_pred_list | 得到 model 对 buck_iter 的预测,返回`2dlist`
get_pred_pd   | 得到 model 对 buck_iter 的预测,返回`pd`
creterion_val | 输入模型的预测值(2dlist或df),输出对指定tsv文件的预测分数