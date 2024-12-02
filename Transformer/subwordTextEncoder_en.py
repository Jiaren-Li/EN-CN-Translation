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
#耗时任务
with codecs.open(DIR_PATH+'/train.en', 'r', 'utf-8') as f:
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