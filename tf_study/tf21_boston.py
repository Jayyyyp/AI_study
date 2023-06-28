import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time, datetime

# 1. 데이터

path = './data/boston/'
x_train = pd.read_csv(path + 'train-data.csv', index_col=0)
x_test = pd.read_csv(path + 'train-target.csv', index_col=0)
y_train = pd.read_csv(path + 'test-data.csv', index_col=0)
y_test = pd.read_csv(path + 'test-target.csv', index_col=0)

print(x_train.shape,x_test.shape)   # (333, 11) (333, 0)
print(y_train.shape,y_test.shape)   # (173, 11) (173, 0)
print(x_train.columns) # Index(['zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'lstat'],dtype='object')
print(y_train.columns) # Index(['zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'lstat'],dtype='object')
print(x_train.describe)
print(x_train.info())

# scaler 적용

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성

model = Sequential()
model.add(Dense())

# 3. 컴파일, 훈련

# 4. 평가, 예측
