import csv
import numpy as np

# 날짜,시간,수평일사량,경사일사량,외기온도,모듈온도,VCB출력,ACB출력,인버터츨력

list = np.loadtxt("finishData.csv", delimiter=',')

dataX = list
dataY = list[:,[-1]]

for i in range(24*2):
    print(dataX[i])

print("==========")

for i in range(24*2):
    print(dataY[i])