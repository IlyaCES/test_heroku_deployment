import os
import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from model.model import Model


app = Flask(__name__)
api = Api(app)

model = Model()


class Predict(Resource):
  @staticmethod
  def post():
    data = request.get_json()

    X = [data[v] for v in '0,1,2,3,4,5,6,7,8,9'.split(',')]

    prediction = model.predict(X)

    return jsonify({
      'Class': int(prediction)
    })
    

api.add_resource(Predict, '/predict')


if __name__ == '__main__':
    app.run(debug=True)