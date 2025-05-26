import tensorflow as tf
from poetry_sentiment_project.poetry_generator.dataset import tokenizer
from poetry_sentiment_project.poetry_generator import settings
from poetry_sentiment_project.poetry_generator import utils

model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)

print(utils.generate_random_poetry(tokenizer, model))
print(utils.generate_random_poetry(tokenizer, model, s='床前明月光，'))
print(utils.generate_acrostic(tokenizer, model, head='海阔天空'))
from poetry_sentiment_project.poetry_generator.dataset import Tokenizer

tokenizer = Tokenizer.from_file("poetry_sentiment_project/poetry_generator/tokenizer.json")

