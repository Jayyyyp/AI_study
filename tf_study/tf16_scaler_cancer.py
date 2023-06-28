import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y= datasets.target

print(x.shape)  # (569, 30)
print(y.shape)  # (569, )
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, x_test.shape)   # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)   # (398, ) (171, )

# Scaler 적용
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(68, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))  
    # 이진분류는 마지막 아웃풋 레이어에 무조건 sigmoid 함수 사용

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=128)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)

# loss :  0.4129038453102112
# mse :  0.06722506880760193
# accuracy :  0.9239766001701355

# Standard Scaler
# loss :  0.27322760224342346
# mse :  0.03468601033091545
# accuracy :  0.9590643048286438

# Minmax Scaler
# loss :  0.37110435962677
# mse :  0.0553257130086422
# accuracy :  0.9356725215911865

# MaxAbsScaler
# loss :  0.17261578142642975
# mse :  0.03956836462020874
# accuracy :  0.9532163739204407

# RobustScaler
# loss :  0.2294492870569229
# mse :  0.03839480131864548
# accuracy :  0.9473684430122375