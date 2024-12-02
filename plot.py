import tensorflow_datasets as tfds
import tensorflow as tf
import jieba
import codecs
import collections
import sys
from operator import itemgetter
import tqdm

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DIR_PATH="/home/xujingzhou/en-zh/"

len_en=[]
with codecs.open(DIR_PATH+"train.en","r","utf-8") as f:
    for line in f:
        line = line[:-1]
        if len(line)<700:
            len_en.append(len(line))
        
len_zh=[]
with codecs.open(DIR_PATH+"train.zh","r","utf-8") as f:
    for line in f:
        line = line[:-1]
        if len(line)<500:
            len_zh.append(len(line))

        
df=pd.DataFrame(list(zip(len_en,len_zh)), columns = ['en','zh'])



plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus']=False

# 绘制核密度图
df.zh.plot(kind = 'kde', color = 'red')


# 绘制直方图
df.zh.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black',density=True)

plt.savefig("zh.png")

# 绘制核密度图
df.en.plot(kind = 'kde', color = 'red')


# 绘制直方图
df.en.plot(kind = 'hist', bins = 20, color ='lawngreen' , edgecolor = 'black',density=True)

plt.savefig("en.png")
            


