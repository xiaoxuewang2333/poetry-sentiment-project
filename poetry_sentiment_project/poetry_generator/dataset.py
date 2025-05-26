from collections import Counter
import math
import numpy as np
import tensorflow as tf
import json
from poetry_sentiment_project.poetry_generator import settings


class Tokenizer:
    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.token_dict_rev = {v: k for k, v in token_dict.items()}
        self.vocab_size = len(token_dict)

    def token_to_id(self, token):
        return self.token_dict.get(token, self.token_dict['[UNK]'])

    def id_to_token(self, token_id):
        return self.token_dict_rev[token_id]

    def encode(self, text):
        token_ids = [self.token_to_id('[CLS]')]
        for ch in text:
            token_ids.append(self.token_to_id(ch))
        token_ids.append(self.token_to_id('[SEP]'))
        return token_ids

    def decode(self, token_ids):
        spec_tokens = {'[CLS]', '[SEP]', '[PAD]'}
        return ''.join([self.id_to_token(i) for i in token_ids if self.id_to_token(i) not in spec_tokens])

    @staticmethod
    def from_file(path):
        with open(path, "r", encoding="utf-8") as f:
            token_dict = json.load(f)
        return Tokenizer(token_dict)

    def to_file(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_dict, f, ensure_ascii=False, indent=2)


# 加载原始数据
disallowed_words = settings.DISALLOWED_WORDS
max_len = settings.MAX_LEN
min_freq = settings.MIN_WORD_FREQUENCY

with open(settings.DATASET_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [line.replace("：", ":") for line in lines]

poetry = []
for line in lines:
    if line.count(":") != 1:
        continue
    _, last_part = line.split(":")
    if any(w in last_part for w in disallowed_words):
        continue
    if len(last_part) > max_len - 2:
        continue
    poetry.append(last_part.strip())

# 构建词表
counter = Counter()
for sentence in poetry:
    counter.update(sentence)

_tokens = [(token, count) for token, count in counter.items() if count >= min_freq]
_tokens = sorted(_tokens, key=lambda x: -x[1])
_tokens = [token for token, _ in _tokens]
_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + _tokens
token_dict = dict(zip(_tokens, range(len(_tokens))))

tokenizer = Tokenizer(token_dict)

np.random.shuffle(poetry)


class PoetryDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, random=False):
        self.data = data
        self.random = random
        self.steps = len(data) // settings.BATCH_SIZE

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        batch_data = self.data[index * settings.BATCH_SIZE : (index + 1) * settings.BATCH_SIZE]
        x, y = [], []
        for line in batch_data:
            token_ids = tokenizer.encode(line)
            if len(token_ids) < 2:
                continue
            x.append(token_ids[:-1])
            y.append(token_ids[1:])
        x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=settings.MAX_LEN, padding='post')
        y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=settings.MAX_LEN, padding='post')
        return x, tf.keras.utils.to_categorical(y, num_classes=tokenizer.vocab_size)

    def for_fit(self):
        return self

# 允许外部调用
__all__ = ["PoetryDataGenerator", "poetry", "tokenizer"]
