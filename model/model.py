import dvc
import joblib


class Model:
  def __init__(self):
    with dvc.api.open('models/m.model') as f:
      self.model = joblib.load(f)

  def predict(self, X):
    if self.model is None:
     raise Exception('Model not trained yet')

    prediction = self.model.predict([X])
    return prediction[0]