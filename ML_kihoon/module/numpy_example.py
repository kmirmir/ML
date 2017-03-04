import numpy

# numpy.random.rand is make the list like matrix
randomList = numpy.random.rand(7,2)
print(randomList)
print("----------")
# index is select the list value
print(randomList[6])
print("----------")
print(numpy.random.rand(4, 3))
print("----------")
inputList = [1, 2]

# numpy.asarray() is change the list value to matrix
array = numpy.asarray(inputList)
print(array)
print(array[0])