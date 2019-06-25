#0.사용할 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#랜덤시드 고정시키기
np.random.seed(5)

#1.데이터 준비하기
dataset = pd.read_csv('movie_weather')
dataset = dataset.values
dataset = dataset.astype('float32')  # float형

#2.데이터셋 생성하기

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(1e-8, 1 - 1e-8))
x_train_0 = dataset[:, 0]
x_train_0 = x_train_0[:, None]
x_train_0 = scaler.fit_transform(x_train_0)
x_train_1 = dataset[:, 1]
x_train_1 = x_train_1[:, None]
x_train_1 = scaler.fit_transform(x_train_1)
x_train_2 = dataset[:, 2]
x_train_2 = x_train_2[:, None]
x_train_2 = scaler.fit_transform(x_train_2)
x_train_3 = dataset[:, 3]
x_train_3 = x_train_3[:, None]
x_train_3 = scaler.fit_transform(x_train_3)
x_train_4 = dataset[:, 4]
x_train_4 = x_train_4[:, None]
x_train_4 = scaler.fit_transform(x_train_4)
x_train_5 = dataset[:, 5]
x_train_5 = x_train_5[:, None]
x_train_5 = scaler.fit_transform(x_train_5)
x_train_6 = dataset[:, 6]
x_train_6 = x_train_6[:, None]
x_train_6 = scaler.fit_transform(x_train_6)
x_train_7 = dataset[:, 7]
x_train_7 = x_train_7[:, None]
x_train_7 = scaler.fit_transform(x_train_7)
x = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6, x_train_7], axis=1)

y = dataset[:,4]
x = dataset[:,:4]

x_train = x[0:156,]
y_train = y[0:156,]
x_val = x[156:208]
y_val = y[156:208]
x_test = x[208:,]
y_test = y[208:,]

#3.모델 구성하기
model = Sequential()
model.add(Dense(20, input_dim=4, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

#4.모델 학습과정 설정하기
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#5.모델 학습시키기
hist = model.fit(x_train, y_train, epochs=500, batch_size = 100, validation_data=(x_val, y_val))

#6.모델 평가하기
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print(hist.history['loss'])

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

#6-1. 에포크와 loss, val_loss, acc, val_acc 그래프
loss_ax.plot(hist.history['loss'],'y', label='train loss')
loss_ax.plot(hist.history['val_loss'],'r', label='val loss')

#acc_ax,plot(hist.history['acc'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_acc'], 'g', label ='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')

loss_ax.legend(loc='upper left')

plt.show()



# 7. 예측하기
y_predict = model.predict(x_test)
df = pd.DataFrame(y_predict)
df.insert(1,'y_test',y_test)
df.to_csv("predict_regression_movie.csv")

#7-2. RMSE 구하기
print(np.sqrt(mean_squared_error(y_test, y_predict)))

# 7-3. 예측값과 실제값 그래프
plt.plot(y_test, label='y_test')
plt.plot(y_predict, label='y_predict')
plt.legend()
plt.show()