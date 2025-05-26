import numpy as np
from poetry_sentiment_project.poetry_generator import settings


def generate_random_poetry(tokenizer, model, s=''):
    token_ids = tokenizer.encode(s)
    token_ids = token_ids[:-1]  # 移除末尾 [SEP]

    while len(token_ids) < settings.MAX_LEN:
        inputs = np.array([token_ids])
        predictions = model.predict(inputs)[0]
        last_token_logits = predictions[-1]

        # 只保留有效词范围，排除[PAD], [CLS], [SEP]
        probs = last_token_logits[3:]
        p_args = probs.argsort()[-settings.TOP_K:][::-1]
        p = probs[p_args]
        p = p / np.sum(p)

        next_token = np.random.choice(p_args, p=p) + 3  # 补回偏移
        token_ids.append(next_token)

        # 如果遇到 SEP 说明诗已结束
        if next_token == tokenizer.token_to_id('[SEP]'):
            break

    # 去除前后的特殊 token
    text = ''.join([tokenizer.id_to_token(tid)
                    for tid in token_ids if tid > 3])
    return text


def generate_acrostic(tokenizer, model, head):
    token_ids = tokenizer.encode('')[:-1]
    punctuations = ['，', '。']
    punctuation_ids = {tokenizer.token_to_id(p) for p in punctuations}
    poetry = []
    for ch in head:
        poetry.append(ch)
        token_ids.append(tokenizer.token_to_id(ch))
        while True:
            output = model(np.array([token_ids], dtype=np.int32))
            _probas = output.numpy()[0, -1, 3:]
            p_args = _probas.argsort()[::-1][:100]
            p = _probas[p_args]
            p = p / sum(p)
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index] + 3
            token_ids.append(target)
            if target > 3:
                poetry.append(tokenizer.id_to_token(target))
            if target in punctuation_ids:
                break
    return ''.join(poetry)

def format_poetry_line(poem: str, line_length: int = 5) -> str:
    """
    将诗句格式化为四行，每行 line_length 个字
    """
    return '\n'.join([poem[i:i+line_length] for i in range(0, len(poem), line_length)])
