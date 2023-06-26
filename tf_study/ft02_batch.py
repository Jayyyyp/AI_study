# 1. 데이터
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
# mae loss : 0.00307

result = model.predict([6])
print('6의 예측값 : ', result)
# mae 6의 예측값 : 6.0173