import tensorflow as tf
import numpy as np
from dataset import load_data

_, _, _, _, tokenizer = load_data()
model = tf.keras.models.load_model('sentiment_model.h5')

def predict_sentiment(poem, max_len=32):
    seq = tokenizer.texts_to_sequences([poem])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    model = tf.keras.models.load_model('sentiment_model.h5')
    pred = model.predict(padded)[0]
    labels = ['positive', 'neutral', 'negative']
    for i, prob in enumerate(pred):
        print(f"{labels[i]}: {prob:.4f}")
    return labels[np.argmax(pred)]

print(predict_sentiment("月云城似寒，春此门酒香。"))
