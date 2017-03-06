import numpy as np

file_data = np.loadtxt('training_one_variable.txt', unpack=True, dtype='float32')
print(file_data)

print(file_data[0:-1])
print(file_data[-1])