import dvc.api
import joblib
import os
from subprocess import check_output

class Model:
  def __init__(self):
    with dvc.api.open('models/m.model', mode='rb') as f:
      self.model = joblib.load(f)

  def predict(self, X):
    if self.model is None:
     raise Exception('Model not trained yet')

    prediction = self.model.predict([X])
    return prediction[0]