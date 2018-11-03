# seedcup2018

[主页](https://uniqueai.me/seedcup2018/)

[主要参考](https://github.com/wabyking/TextClassificationBenchmark)

关于[种子杯复赛](http://rank.dian.org.cn/static/index.html)的代码, 总行数300行左右

暂时没有训练数据

## env prerequisite

+ 不限平台
+ python 3.*(以下没有特别说明,均为2018-10最新)

  + numpy
  + pandas
  + sklearn
  + pytorch(pytorch-cpu allowed)
    + torchtext
  + gensim

## 数据概览

![data.png](visualization.png)

## 目录结构

    .                       
    ├── answerlastlong.txt  预测后的文件
    ├── config.py           
    ├── data                处理后的数据
    │   ├── test_w.tsv      
    │   ├── train_w.tsv     
    │   ├── val_w.tsv       
    │   └── w300.txt        词向量模型
    ├── datahelper.py       
    ├── dime300             某个保存训练结果的文件夹
    │   └── 13.pth          用来载入的模型
    ├── main.py             主程序
    ├── model               
    │   └── BNBLSTMr.py     
    ├── raw
    │   └── ...          比赛方提供的数据
    ├── train.py            提供训练代码支持
    └── util.py             提供数据处理支持


## 运行

如果缺失data数据,用 `py datahelper.py` 生成(需要`raw/`文件夹的原始数据)

运行时首先保证你当前的目录结构与上面一致

假设你要保存模型在 `abc` 文件夹

`py main.py abc`

### 载入模型

修改 `config.py` 里面的 `LAST_EPOCH` 参数 ; 并把 `LOADMODEL` 参数设置为 模型路径

 `py main.py abc`

### 手动修改答案

使用 ipython 逐条运行 `main.py` (其实也就5行)

`ans = util.get_pred_list(model, test_iter, use_pandas=True)`

此处返回值为 `pd.DataFrame` 格式, 可以手动修改

## 参数接口
