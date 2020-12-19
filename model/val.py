import joblib
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix


model_path = 'models/m.model'

try:
  model = joblib.load(model_path)
except:
  sys.exit(f"Can't find model {model_path}")


df = pd.read_csv('dataset/data_test.csv')
X = df.drop('y', axis=1)
y_true = df['y']

y_pred = model.predict(X)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

with open("model/metrics.json", 'w') as outfile:
        json.dump({'accuracy': accuracy, 'f1_score': f1}, outfile)

cm = plot_confusion_matrix(model, X , y_true)
plt.savefig('model/confusion_matrix.png')