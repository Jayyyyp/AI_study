import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)         # (150, 4)(150, )
print(datasets.feature_names)   
    # ['sepal length (cm)', 'sepal width(cm)',]
print(datasets.DESCR)

'''
###### 원핫인코딩 one-hot-encoding ######
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)
'''

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape)  # (105, 4) (45, 4) => 4는 input
print(y_train.shape, y_test.shape)  # (105,) (45,) => 3은 output

# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['mse', 'accuracy'])
start_time = time.time()
model.fit(x_train, y_train,
          epochs=500, batch_size=32)
end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

# 4. 평가, 예측
loss, mse, accuracy =  model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)