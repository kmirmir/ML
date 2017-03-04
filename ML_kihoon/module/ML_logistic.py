import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


xy = np.loadtxt('logistic.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]
print(x_data)
print(y_data)

weight_val = []
cost_val = []

weight = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

h = tf.matmul(weight, x_data)
hypothesis = tf.div(1., 1. + tf.exp(-h))
cost = -tf.reduce_mean(y_data * tf.log(hypothesis) + (1 - y_data) * tf.log(1 - hypothesis))

learning_rate = 0.7

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    weight_val.append(sess.run(weight))
    cost_val.append(sess.run(cost))

    if step % 20 == 0 :
        print(step, sess.run(cost), sess.run(weight))

print(weight)
print(weight.get_shape())
print(weight_val)

# print('---------------------------')
#
# print(sess.run(hypothesis, feed_dict={x: [[1], [2], [2]]}) > 0.5)
# print(sess.run(hypothesis, feed_dict={x: [[1], [5], [5]]}) > 0.5)
# print(sess.run(hypothesis, feed_dict={x: [[1, 1], [4, 3], [3, 5]]}) > 0.5)
#
#
# plt.plot(weight_val, cost_val, 'ro')
# plt.plot(weight,cost,'ro')
# plt.legend()
# plt.show()