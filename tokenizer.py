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

time_start = time.time()
DIR_PATH="/home/xujingzhou/en-zh/"#好像得copy到自己的目录下面#copy了已经

vocab_zh=[]
with codecs.open(DIR_PATH+"train.zh","r","utf-8") as f:
    for line in f:
        words=jieba.cut(line)
        for word in words:
            vocab_zh.append(word)


vocab_zh=list(set(vocab_zh))
#print("vocab len:",len(vocab_zh))
#vocab_zh.sort(key = lambda i:len(i),reverse=True) 
#vocab_zh

vocab_en=[]
with codecs.open(DIR_PATH+"train.en","r","utf-8") as f:
    for line in f:
        words=jieba.cut(line)
        for word in words:
            vocab_en.append(word)



vocab_en=list(set(vocab_en))
#print("vocab len:",len(vocab_en))
#vocab_en.sort(key = lambda i:len(i),reverse=True) 
#vocab_en

vocab=vocab_zh+vocab_en
vocab=set(vocab)
with codecs.open("/home/xujingzhou/transformer-ljr/vocab.enzh","w","utf-8") as f:
    for word in vocab:
        f.write(word+"\n")

print("耗时  ",time.time-time_start)
