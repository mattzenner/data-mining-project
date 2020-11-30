from keras.datasets.mnist import load_data
import pandas as pd
from sklearn.model_selection import train_test_split


aslX, _ = train_test_split(pd.read_csv('D://gan//asl_mnist_style.csv'))
X = load_data()

print('MNIST')
print('type: ', type(X))
print(X[0], X[1])
print('len X:', len(X))
print(len(X[0]), len(X[1]))

print()
print('ASL')
print('type: ', type(aslX))
#print(aslX[0], aslX[1])
print('len aslX:', len(aslX))
print(len(aslX[0]), len(aslX[1]))
