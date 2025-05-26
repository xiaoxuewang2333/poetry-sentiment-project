import os
import gradio as gr
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import json

from poetry_sentiment_project.poetry_generator import settings, utils
from poetry_sentiment_project.poetry_generator.dataset import Tokenizer
from poetry_sentiment_project.sentiment_analysis.dataset import load_data

# 加载 tokenizer
with open(settings.TOKENIZER_PATH, "r", encoding="utf-8") as f:
    token_dict = json.load(f)
tokenizer = Tokenizer(token_dict)

# 加载模型
poem_model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)
_, _, _, _, sentiment_tokenizer = load_data()
sentiment_model = tf.keras.models.load_model(settings.SENTIMENT_MODEL_PATH)

# 历史记录
history_records = []

def generate_poem(mode, input_text, acrostic_head):
    if mode == "随机生成":
        poem = utils.generate_random_poetry(tokenizer, poem_model)
    elif mode == "续写":
        poem = utils.generate_random_poetry(tokenizer, poem_model, s=input_text)
    elif mode == "藏头诗":
        poem = utils.generate_acrostic(tokenizer, poem_model, head=acrostic_head)
    else:
        poem = "无效模式"

    history_records.append({
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'source': '生成',
        'poem': poem,
        'predicted': '',
        'positive': '',
        'neutral': '',
        'negative': ''
    })
    return poem

def predict_sentiment(poem):
    seq = sentiment_tokenizer.texts_to_sequences([poem])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=32, padding='post')
    pred = sentiment_model.predict(padded)[0]
    labels = ['positive', 'neutral', 'negative']
    result = {labels[i]: float(f"{prob:.4f}") for i, prob in enumerate(pred)}
    top_label = labels[np.argmax(pred)]
    history_records.append({
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'source': '情感分析',
        'poem': poem,
        'predicted': top_label,
        'positive': result['positive'],
        'neutral': result['neutral'],
        'negative': result['negative']
    })
    return top_label, result

def get_history():
    return pd.DataFrame(history_records)

def export_history():
    df = pd.DataFrame(history_records)
    df.to_csv("history_export.csv", index=False)
    return "history_export.csv", df

def clear_history():
    history_records.clear()
    return get_history()

with gr.Blocks() as demo:
    gr.Markdown("# 📜 古诗生成与情感分析系统")

    with gr.Tab("🖋️ 古诗生成"):
        mode = gr.Dropdown(["随机生成", "续写", "藏头诗"], label="生成模式")
        input_text = gr.Textbox(label="前半句")
        acrostic_head = gr.Textbox(label="藏头字")
        generate_btn = gr.Button("生成")
        poem_output = gr.Textbox(label="生成的古诗")
        generate_btn.click(generate_poem, inputs=[mode, input_text, acrostic_head], outputs=poem_output)

    with gr.Tab("❤️ 情感分析"):
        poem_input = gr.Textbox(label="输入古诗")
        predict_btn = gr.Button("分析")
        result_label = gr.Textbox(label="情绪类别")
        result_probs = gr.Label(label="情绪概率")
        predict_btn.click(predict_sentiment, inputs=poem_input, outputs=[result_label, result_probs])

    with gr.Tab("📊 历史记录"):
        table = gr.Dataframe(get_history, label="记录")
        export_btn = gr.Button("导出 CSV")
        clear_btn = gr.Button("清空")
        export_btn.click(export_history, outputs=[gr.File(), table])
        clear_btn.click(clear_history, outputs=table)

if __name__ == '__main__':
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)


