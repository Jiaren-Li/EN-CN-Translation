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

DIR_PATH="/home/xujingzhou/en-zh"

time_start = time.time()
with codecs.open(DIR_PATH+'/train.en', 'r', 'utf-8') as f:#这里读词表会不会好一点呢？
    # time_start=time.time()
    wordText = f.readlines()
    # print("耗时：",time.time()-time_start)
    #16s
    #将英文语料中的标点符号和英文字母转换为小写
    wordText = [line.lower().strip() for line in wordText]
    # print("耗时：",time.time()-time_start)
    #去除英文语料中的标点符号
    wordText = [''.join(c for c in s if c not in ('!', '.', '?', ':', ';', ',')) for s in wordText]
    # print("耗时：",time.time()-time_start)
    # print(text)
    encoder=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(wordText, target_vocab_size=2**15)

encoder.save_to_file(DIR_PATH+'/subwordTextEncoder.en')
print("耗时  ",time.time-time_start)
#耗时任务
# with codecs.open(DIR_PATH+'/train.zh','r','utf-8') as f:
#     wordText = f.readlines()
#     wordText=[line.lower().strip() for line in wordText]
#     wordText=[''.join(c for c in s if c not in ('！', '.', '？', '：', '；', ',','，','。','、')) for s in wordText]
#     #中文分词
#     # text_zh=[list(jieba.cut(line.strip())) for line  in text_zh]
#     encoder=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(wordText,target_vocab_size=2**15)#好像这个东西足够强？？##但是这个东西似乎空间复杂度太高了
##上面这个运行不完,会炸内存
##改为读词表来做吧
with codecs.open(DIR_PATH+"/vocab2.zh",'r','utf-8') as f:#但词表质量堪忧啊
    vocab=f.readlines()
    vocab=[line.lower().strip() for line in vocab]
    vocab=[''.join(c for c in s if c not in ('！', '.', '？', '：', '；', ',','，','。','、')) for s in vocab]
    encoder=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(vocab,target_vocab_size=2**15)

encoder.save_to_file(DIR_PATH+'/subwordTextEncoder.zh')

print("耗时：",time.time()-time_start)