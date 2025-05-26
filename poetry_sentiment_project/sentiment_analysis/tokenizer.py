from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer(texts, num_words=3000):
    tokenizer = Tokenizer(num_words=num_words, oov_token='[UNK]')
    tokenizer.fit_on_texts(texts)
    return tokenizer
