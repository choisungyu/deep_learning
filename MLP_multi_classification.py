#0. 패키지
from bokeh.models import pd
from sympy.printing.tests.test_numpy import np

# 랜덤시드 고정
np.random.seed(5)

#1. 데이터 준비하기
dataset = pd.read.csv('Third_degree_scald_2.jpg')
dataset = dataset.values

#2. 데이터셋 생성하기
y = dataset[:,4]
x = dataset[:,:4]

