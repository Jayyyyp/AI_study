# 1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

    # numpy 배열로 각각 [1,2,3]을 갖는 x와 y 변수 선언

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
    # keras 라이브러리를 사용하여 인공 신경망 모델을 구성 

model = Sequential()        # 객체는 레이어를 선형으로 연결하여 구성(모델이 순차적으로 진행된다는 의미)
    # model.add(Dense())는 완전 연결 레이어를 추가하는 명령이다. 
    # Dense는 모든 뉴런이 이전 레이어의 모든 뉴런과 연결되어있는 레이어이다.

model.add(Dense(3, input_dim = 1))      # 입력층
    # 입력 레이어는 input_dim = 1 파라미터를 통해 1개의 입력을 받는다고 지정되어 있다.

model.add(Dense(5))                     # 히든레이어1
model.add(Dense(6))                     # 히든레이어2
model.add(Dense(6))                     # 히든레이어3
model.add(Dense(4))                     # 히든레이어4

model.add(Dense(1))                     # 출력층
    # 출력 레이어는 1개의 뉴런을 가지며, 이 뉴런이 최종 예측값을 출력한다.


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
    # 손실 함수 mse를 최적화 알고리즘으로 adam을 사용하여 메소드는 모델을 컴파일한다.
    # mse는 실제 값과 예측 값의 차이의 제곱의 평균을 나타내는 손실함수이다.
    # mae는 실제값과 예측값의 차이의 절대값을 평균하는 방법이다.

    # mse가 큰 오차에 대해 mae보다 더 큰 패널티를 부과한다.
    # 큰 오차를 피하고싶다면, mse를 사용하고, 모든 오차를 동등하게 취급하려면 mae를 사용해야 한다.
    # adam은 확률적 경사 하강법을 기반으로 한 최적화 알고리즘이다. 

model.fit(x, y, epochs=100)
    # 메소드는 모델을 훈련한다.
    # x와 y를 사용하여 100회의 에포크 훈련을 설정한다.
    # 에포크란 전체 데이터셋에 대해 한 번 학습을 완료하는 주기를 의미한다.

# 4. 평가, 예측
loss = model.evaluate(x, y)
    # 모델의 성능을 평가한다.
    # x와 y를 사용하여 손실값을 반환한다.(mse 손실 함수에 의해 계산된 값)

print('loss : ', loss)                  # 0.00063

result = model.predict([4])
    # 새로운 입력 데이터에 대한 예측을 수행한다.
print('result : ',result)               # 3.9996