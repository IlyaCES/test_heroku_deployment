from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('dataset/data.csv')
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

train_df = pd.DataFrame(X_train)
train_df['y'] = y_train

test_df = pd.DataFrame(X_test)
test_df['y'] = y_test

train_df.to_csv('dataset/data_train.csv', index=False)
test_df.to_csv('dataset/data_test.csv', index=False)