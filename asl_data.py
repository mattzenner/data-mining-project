import pandas as pd

data = pd.read_csv('D://data_mining_project//training_set_2//training_set//train.csv')

print(data.head())
print(len(data))
print(data[:5])
print(28.0 in data['label'])
print(12.0 in data['label'])
