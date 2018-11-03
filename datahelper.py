from config import *
from gensim.models import Word2Vec
import pandas as pd
import os
# based on jupyter notebook

# get AllData
allWordDflist=[]
for rawfilename in [OTRAIN,OVAL,OTRAIN_A,OVAL_A,OTEST,OTEST_A]:
    allWordDflist.append(pd.read_csv(rawfilename,sep='\t')
                 .drop(columns=['item_id','title_characters','description_characters'])
                 .rename(index=str,columns={'title_words':"tw",'description_words':'dw'})
                )
    
# get all word Data
allWordDf=pd.concat(
    allWordDflist,ignore_index=True,sort=False
    )#.sample(frac=1)
## simply delete repeat
allWordDf.drop_duplicates(inplace=True)
allWordDf.reset_index(inplace=True,drop=True)

## get word matrix

WwordMat=[x.split(',') for x in (allWordDf['tw']+','+allWordDf['dw'])]

Wmodel = Word2Vec(WwordMat,
                  size = 300, workers= THREAD, 
                  sample=1e-3,
                  sg=1, hs=1, # skip-gram & hierarchical softmax
                  #,trim_rule=None
                 ) 
Wmodel.wv.save_word2vec_format("data/w300_a.txt")

### also, you can use stop words

def my_trim_rule(word,count,min_count):
   return gensim.utils.RULE_DISCARD if word in ['w27','w2'] else gensim.util.RULE_DEFAULT

# get all data with label without repetition

allLabeledDf=pd.concat(
    allWordDflist[:4],ignore_index=True,sort=False
    ).sample(frac=1,random_state=201) # shuffle AllLabeled Df

allLabeledDf.drop_duplicates(inplace=True)
allLabeledDf.reset_index(drop=True,inplace=True) 

## preprocess and save

allLabeledDf['w']=allLabeledDf['tw']+','+allLabeledDf['dw']
allLabeledDf.drop(columns=['dw','tw'],inplace=True)
allLabeledDf=allLabeledDf[allLabeledDf.columns[[3,0,1,2]]]

train=allLabeledDf.sample(frac=0.9)
val=allLabeledDf.drop(train.index) # without shuffle
testdf=pd.read_csv(OTEST,sep='\t')

train.to_csv('data/train_w.csv',index=False,sep='\t')
val.to_csv('data/val_w.tsv',index=False,sep='\t')
testdf.to_csv('data/test_w.tsv',index=False,sep='\t')


