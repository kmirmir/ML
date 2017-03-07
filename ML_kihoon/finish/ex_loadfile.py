import numpy as np

file_data = np.loadtxt('training_multi_variable.txt', unpack=True, dtype='float32')
print(file_data)

print(file_data[0:-1])
print(file_data[-1])

print("======")

import tensorflow as tf
# 에러 내용 : typeError: DataType float32 for attr 'T' not in list of allowed values: int32, int64
# 에러 내용 : ValueError: initial_value must have a shape specified: Tensor("random_uniform:0", shape=(?, ?), dtype=float32)
# 위에거 해결함. 텐서 안에 있는 값을 가져오는 코드로 해결
x = file_data[0:-1]
y = file_data[-1]

print(x)
print(y)

len_data = tf.Variable(len(x))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(len_data)
print(len_data.get_shape())
#텐서모양
print(len_data.value())
#이게 텐서 안에 있는 값을 가져오는 코드임
print(len_data.eval(sess))
print(len(x))
# print(sess.run(weight, feed_dict={len_data: len(x)}))




weight = tf.Variable(tf.random_uniform([1,len_data.eval(sess)], -1.0, 1.0))

