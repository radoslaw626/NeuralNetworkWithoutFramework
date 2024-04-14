import pandas as pd
from sklearn import preprocessing

#reading data from file
data = pd.read_csv("train.csv")
print(data.shape)
print(data.head().to_string())

#checking if exists records with empty values
print(data.isnull().sum())
print(data.describe().to_string())

resultColumn=pd.DataFrame(data, columns=['class'])
data = data.drop(['class', 'id'], axis = 1)

#scaling values with MinMax scalar (values 0-1)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
names = data.columns
d = scaler.fit_transform(data)
scaledDb = pd.DataFrame(d, columns=names)

result = pd.concat([scaledDb, resultColumn], axis=1)

print(result.to_string())

#saving normalised dataset to new .csv file
result.to_csv("NormalizedDb.csv")
