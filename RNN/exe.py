import numpy as np

filename = 'exe.csv'
data = np.loadtxt(filename, delimiter=',')
x = data
y = data[:, [-1]]  # Close as label

print(x)
print(y)

dataX = data[:3]
dataY = data[:3 ,[-1]]
print(dataX)
print(dataY)