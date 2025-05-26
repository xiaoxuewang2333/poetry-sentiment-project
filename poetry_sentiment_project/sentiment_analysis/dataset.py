import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from poetry_sentiment_project.sentiment_analysis.tokenizer import create_tokenizer

def load_data(path = r"poetry_sentiment_project/sentiment_analysis/data/poem_sentiment.csv", max_len=32):
    df = pd.read_csv(path)
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    df['label'] = df['label'].map(label_map)

    x_train, x_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    tokenizer = create_tokenizer(x_train)
    
    x_train_seq =tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_len, padding='post')
    x_val_seq = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=max_len, padding='post')
    
    y_train_cat =tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val_cat =tf.keras.utils.to_categorical(y_val, num_classes=3)
    
    return x_train_seq, y_train_cat, x_val_seq, y_val_cat, tokenizer
