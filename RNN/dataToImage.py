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

path = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/'
data1_path = 'dataToDay'
data2_path = 'dataToFourHour'
data3_path = 'dataToHour'
data1 = np.loadtxt(path+data1_path+'.csv', delimiter=',')
data2 = np.loadtxt(path+data2_path+'.csv', delimiter=',')
data3 = np.loadtxt(path+data3_path+'.csv', delimiter=',')

# csv에서 데이터 몇번째 거를 가지고 표에 나타낼지 정하는 부분
# x1 = data1[1:100, [3]]
# x2 = data2[1:100, [3]]
x3 = data3[1:150, [3]]
y3 = data3[1:150, [5]]


def showImage(x, x_label, save_file_name):
    # attr = 'o-'  # 선 속성
    plot.plot(x)
    plot.xlabel(x_label)
    plot.savefig(save_file_name+'.png')
    # plot.close('all')
    plot.show()


# showImage(x1, "24 binding", data1_path)
# showImage(x2, "4 binding", data2_path)
showImage(x3, "all 24hour", data3_path+'일사량')
showImage(y3, "all 24hour", data3_path+'온도')