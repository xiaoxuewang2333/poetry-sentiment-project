# 改进版 compare_models.py 内容
import matplotlib.pyplot as plt
import numpy as np

models = ['My LSTM', 'ChatGPT', 'Wenxin']
positive = [10, 8, 6]
neutral = [12, 14, 15]
negative = [8, 6, 9]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(8,6))
ax.bar(x - width, positive, width, label='Positive')
ax.bar(x, neutral, width, label='Neutral')
ax.bar(x + width, negative, width, label='Negative')

ax.set_ylabel('Count')
ax.set_title('各模型生成诗歌的情绪分布对比')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.savefig('improved_emotion_comparison.png', dpi=300)
print('情绪对比图已生成')
