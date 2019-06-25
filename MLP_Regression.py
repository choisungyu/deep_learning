#0.사용할 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
#랜덤시드 고정시키기
np.random.seed(5)

#1.데이터 준비하기
dataset = pd.read_csv('movie_weather.csv')
dataset = dataset.values

#2.데이터셋 생성하기
y = dataset[:,3] #
x = dataset[:,:3]

x_train = x[0:300,] #365개중
y_train = y[0:300,]
x_test = x[300:,]
y_test = y[300:,]


#3.모델 구성하기
model = Sequential()
model.add(Dense(20, input_dim=3, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, activation='linear'))

#4.모델 학습과정 설정하기
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#5.모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 10)

#6.모델 평가하기
scores = model.evaluate(x_test, y_test, batch_size = 10)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print(hist.history['loss'])

import matplotlib.pyplot as plt

fig, loss_ax = matplotlib.pyplot.subplots()

#acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y', label='train loss')
#acc_ax.plot(hist.history['acc'],'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
#acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
#acc_ax.legend(loc='lower left')

matplotlib.pyplot.show()

# 7. 예측하기
y_predict = model.predict(x_test)
df = pd.DataFrame(y_predict)
df.insert(1,'y_test',y_test)
df.to_csv("predict_regression_movie.csv")

#7-2. RMSE 구하기
print(np.sqrt(mean_squared_error(y_test, y_predict)))

# 7-3. 예측값과 실제값 그래프
matplotlib.pyplot.plot(y_test, label='y_test')
matplotlib.pyplot.plot(y_predict, label='y_predict')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()