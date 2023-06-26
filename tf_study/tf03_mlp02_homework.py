import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

x = x.transpose()

# 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 3))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=32)

# 예측 [[10, 1.6, 1]]
loss = model.evaluate(x, y)
print('loss : ', loss)  # loss :  3.877175913657993e-09

result = model.predict([[10, 1.6, 1]])
print('10과 1.6, 1의 예측값 : ', result) # [[19.999971]]