# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : wangxx
# @Time    : 2025/3/14
# @Desc    : 古诗生成模型训练脚本（含 tokenizer 保存）

import tensorflow as tf
import json
import os
from poetry_sentiment_project.poetry_generator.dataset import PoetryDataGenerator, poetry, tokenizer
from poetry_sentiment_project.poetry_generator.model import model
from poetry_sentiment_project.poetry_generator import settings, utils

# ✅ 保存 tokenizer 到 json 文件
tokenizer_json_path = os.path.join("poetry_sentiment_project", "poetry_generator", "tokenizer.json")
with open(tokenizer_json_path, "w", encoding="utf-8") as f:
    json.dump(tokenizer.token_dict, f, ensure_ascii=False, indent=2)
print(f"✅ tokenizer.json 已保存到: {tokenizer_json_path}")

# ✅ 定义回调，展示每个 epoch 后的生成效果
class Evaluate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n---生成诗：---')
        for _ in range(3):
            poem = utils.generate_random_poetry(tokenizer, model)
            print(utils.format_poetry_line(poem))

# ✅ 构造数据生成器
data_generator = PoetryDataGenerator(poetry, random=True)

# ✅ 启动训练
model.fit(
    data_generator.for_fit(),
    steps_per_epoch=data_generator.steps,
    epochs=settings.TRAIN_EPOCHS,
    callbacks=[Evaluate()]
)

# ✅ 保存最终模型
model.save(settings.BEST_MODEL_PATH)
print("✅ 模型训练完成，已保存至:", settings.BEST_MODEL_PATH)
