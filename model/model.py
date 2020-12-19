import dvc.api
import joblib
import os


class Model:
  def __init__(self):
    # with dvc.api.open('models/m.model') as f:
    #   self.model = joblib.load(f)
    # self.model = dvc.api.read('models/m.model')
    try:
      self.model = joblib.load('models/m.model')
    except:
      os.system('dvc pull models/m.model')
      self.model = joblib.load('models/m.model')

  def predict(self, X):
    if self.model is None:
     raise Exception('Model not trained yet')

    prediction = self.model.predict([X])
    return prediction[0]