import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plot

# df = pd.read_csv('./exe.csv')
# df.columns = ['one','two','three','four']
#
# print(df.index)
# print("========")
# print(df.head())
# print("========")
# print(df.info())


data1 = np.loadtxt('/Users/masinogns/PycharmProjects/ML/RNN/dataToDay.csv', delimiter=',')
data2 = np.loadtxt('/Users/masinogns/PycharmProjects/ML/RNN/MyLib/dataToDay.csv', delimiter=',')
data3 = np.loadtxt('/Users/masinogns/PycharmProjects/ML/RNN/MyLib/dataToHour.csv', delimiter=',')

x1 = data1[:, [3]]
x2 = data2[:, [3]]
x3 = data3[:, [3]]


def showImage(x, x_label):
    attr = 'o-'  # 선 속성
    plot.plot(x, attr)
    plot.xlabel(x_label)
    plot.show()


showImage(x1, "4 binding")
# showImage(x2, "24 binding")
# showImage(x3, "all 24hour")