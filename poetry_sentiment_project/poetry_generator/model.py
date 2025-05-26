import tensorflow as tf
from poetry_sentiment_project.poetry_generator.dataset import tokenizer

# 模型结构：输入、嵌入、双层LSTM、输出
poem_input = tf.keras.layers.Input(shape=(None,), name='poem_input')
embedding = tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128)(poem_input)
lstm_1 = tf.keras.layers.LSTM(128, return_sequences=True)(embedding)
lstm_2 = tf.keras.layers.LSTM(128, return_sequences=True)(lstm_1)
output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax'))(lstm_2)

model = tf.keras.models.Model(inputs=poem_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# 查看模型结构
model.summary()
