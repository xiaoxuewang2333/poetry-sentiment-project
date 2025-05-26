import jieba
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 测试用诗句（每组至少2句，确保有重叠字词）
my_model_poems = ["春风又绿江南岸", "江南岸边杨柳青", "杨柳青青江水平"]
chatgpt_poems = ["白日依山尽", "黄河入海流", "欲穷千里目"]
wenxin_poems = ["床前明月光", "疑是地上霜", "举头望明月"]

def tokenize(poems):
    return [" ".join(jieba.cut(p)) for p in poems]

def avg_similarity(poems, label):
    if len(poems) < 2:
        print(f"⚠️ {label} 至少需要两句诗，目前仅 {len(poems)} 句")
        return 0
    texts = tokenize(poems)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)
    upper = sim_matrix[np.triu_indices(len(poems), k=1)]
    avg = np.mean(upper)
    if np.isnan(avg) or avg == 0:
        print(f"⚠️ {label} 计算出的平均相似度为 0，请检查句子是否完全无重叠。")
    return avg

scores = {
    "My LSTM": avg_similarity(my_model_poems, "My LSTM") * 1000,
    "ChatGPT": avg_similarity(chatgpt_poems, "ChatGPT") * 1000,
    "Wenxin": avg_similarity(wenxin_poems, "Wenxin") * 1000,
}

models = list(scores.keys())
values = list(scores.values())

plt.figure(figsize=(8,5))
bars = plt.bar(models, values, color=["#4c72b0", "#55a868", "#c44e52"])
plt.title("各模型生成诗歌的平均文本相似度（放大1000倍）")
plt.ylabel("放大后相似度")

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.5, f"{y:.2f}", ha='center')

plt.ylim(0, max(values) + 10)
plt.tight_layout()
plt.savefig("improved_similarity_result.png", dpi=300)
print("✅ 改进版相似度图已保存为 improved_similarity_result.png")
