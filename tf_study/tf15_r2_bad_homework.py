# [실습]
# 1. r2 score을 음수가 아닌 0.5 이하로 만드세요
# 2. 데이터는 건드리지 마세요
# 3. 레이어는 인풋, 아웃풋 포함 7개 이상 만드세요
#    (히든 레이어가 5개 이상이어야 함)
# 4. batch_size = 1이어야 함
# 5. 히든레이어의 노드(뉴런) 개수는 10 이상 100 이하
# 6. train_size = 0.7
# 7. epochs=100 이상

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

print(x) # 20
print(y) # 20

x_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
y_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

x_test = np.array([15,16,17,18,19,20])
y_test = np.array([15,16,17,18,19,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mse', 'accuracy'])
start_time = time.time()
model.fit(x_train, y_train,epochs=100, batch_size=1)
end_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([21])
print('21의 예측값 : ', result)
