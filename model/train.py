from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib


df = pd.read_csv('dataset/data_train.csv')
X = df.drop('y', axis=1)
y = df['y']


model = DecisionTreeClassifier(max_depth=8).fit(X, y)
joblib.dump(model, 'models/m.model')