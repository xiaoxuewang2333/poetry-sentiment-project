# -*- coding: utf-8 -*-
# @File    : settings.py

# 禁用字符，包含如下的唐诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']

# 每句最大长度（包括起始符）
MAX_LEN = 64

# 最小词频，小于该频率的字将被忽略
MIN_WORD_FREQUENCY = 8

# 每批训练样本数
BATCH_SIZE = 16

# 训练集路径
DATASET_PATH = 'poetry_sentiment_project/poetry_generator/poetry.txt'

# 每个 epoch 结束后展示多少首古诗
SHOW_NUM = 3

# 训练轮数
TRAIN_EPOCHS = 20

# 最佳模型保存路径
BEST_MODEL_PATH = 'poetry_sentiment_project/poetry_generator/best_model.h5'
# Top-K 采样控制：每步只考虑概率前多少个词
TOP_K = 100
TOKENIZER_PATH = "poetry_sentiment_project/poetry_generator/tokenizer.json"

SENTIMENT_MODEL_PATH = "poetry_sentiment_project/sentiment_analysis/sentiment_model.h5"
