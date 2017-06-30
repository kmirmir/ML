import csv
import numpy as np

list = np.loadtxt("output.csv", delimiter=',')
# result = []
#
# for i in range(len(list)):
#     result.append(list[i].replace("\"", ""))
#
# print(result)
# for i in range(len(result)):
#     print(result[i])
# #
# f = open('output.csv', 'w', encoding='utf-8', newline='')
# wr = csv.writer(f)
# for i in range(len(result)):
#     wr.writerow(result[i])
# f.close()

dataX = list
dataY = list[:,[-1]]

for i in range(len(dataX)):
    print(dataX[i])

print("==========")
print("==========")

for i in range(len(dataY)):
    print(dataY[i])