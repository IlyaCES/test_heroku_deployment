from sklearn.datasets import make_classification
import pandas as pd


X, y = make_classification(n_samples=100000, n_features=10,
                          n_informative=7, n_redundant=0,
                          n_classes=5)

df = pd.DataFrame(X)
df['y'] = y

df.to_csv('dataset/data.csv')
print(df.head())