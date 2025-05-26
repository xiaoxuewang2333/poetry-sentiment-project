import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from dataset import load_data

x_train, y_train, x_val, y_val, _ = load_data()
model = load_model("sentiment_model.h5")

y_pred = model.predict(x_val)
y_true = np.argmax(y_val, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'neutral', 'negative'])
disp.plot(cmap=plt.cm.Blues)
plt.title("情绪分类混淆矩阵")
plt.show()
