import numpy as np

a = np.array([1,2,3])
print(a.shape)
print(a)

b = np.array([
    [1,2,3],
    [4,5,6]
])
print(b.shape)
print(b)

txt = np.loadtxt('train.txt')
print(txt)