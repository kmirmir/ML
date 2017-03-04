import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print(x_data)
print(y_data)
print("---")
print(len(x_data))

weight = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))
hypothesis = tf.matmul(weight, x_data)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))


learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


sess.run(train)

for step in range(2001):
    sess.run(train)

    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(weight))