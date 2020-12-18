import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


MODELS_DIR = 'models/'
DATASET_DIR = 'dataset/'


class Model:
  def __init__(self):
    self.df = pd.read_csv(DATASET_DIR + 'Iris.csv')
    try:
      self.model = joblib.load(MODELS_DIR + 'iris.model')
    except:
      self.model = None

  def fit(self):
    self.X = self.df.drop(['Species', 'Id'], axis=1)
    self.y = self.df['Species']
    self.model = DecisionTreeClassifier(max_depth=1).fit(self.X, self.y)
    joblib.dump(self.model, MODELS_DIR + 'iris.model')

  def predict(self, measurement):
    if self.model is None:
      raise Exception('Model not trained yet')
    if len(measurement) != 4:
      raise Exception(f'Expected sepal_length, sepal_width, petal_length, petal_width, got {measurement}')

    prediction = self.model.predict([measurement])
    return prediction[0]


if __name__ == '__main__':
  m = Model()
  m.fit()

  y_true = m.y
  y_pred = m.model.predict(m.X)

  accucary = accuracy_score(y_true, y_pred)

  importances = m.model.feature_importances_

  labels = m.X.columns
  feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
  feature_df = feature_df.sort_values(by='importance', ascending=False)

  sns.set(style="whitegrid")

  axis_fs = 18
  title_fs = 22

  ax = sns.barplot(x="importance", y="feature", data=feature_df)
  ax.set_xlabel('Importance',fontsize = axis_fs) 
  ax.set_ylabel('Feature', fontsize = axis_fs)
  ax.set_title('DecisionTreeClassifier\nfeature importance', fontsize = title_fs)

  plt.tight_layout()
  plt.savefig("model/feature_importance.png",dpi=120) 
  plt.close()

  with open("model/metrics.txt", 'w') as outfile:
    outfile.write(f"Model accuracy {accucary}\n")

  # x = [5.1, 3.5, 1.4, 0.2]
  # print('Predict for ' + str(x))
  # print('Prediction ' + str(m.predict(x)))
