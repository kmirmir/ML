import numpy as np
import matplotlib.pyplot as plt

# 데이터 랜덤하게 생성하기.
num_points = 100
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()

import tensorflow as tf

weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))
print(bias.get_shape())
y = weight*x_data+bias

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss=loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(10):
    sess.run(train)

    print(step, sess.run(weight), sess.run(bias))
    print(step, sess.run(loss))

    # #그래프 표시
    # plt.plot(x_data, y_data, 'ro')
    # plt.plot(x_data, sess.run(weight)*x_data + sess.run(bias))
    # plt.xlabel('x')
    # plt.xlim(-2,2)
    # plt.ylabel('y')
    # plt.ylim(0.1,0.6)
    # plt.legend()
    # plt.show()
    #
