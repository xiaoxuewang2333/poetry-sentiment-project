
import tensorflow as tf
from dataset import load_data
from model import create_model

x_train, y_train, x_val, y_val, tokenizer = load_data()
vocab_size = len(tokenizer.word_index) + 1
input_length = x_train.shape[1]

model = create_model(vocab_size, input_length)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=callbacks
)

model.save('sentiment_model.h5')
print("✅ 模型训练完成并保存为 sentiment_model.h5")
