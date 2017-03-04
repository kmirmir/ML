import tensorflow as tf
from matplotlib import pyplot as plt

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that computer is compute y_data = weight * x_data + b
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# my hypothesis
hypothesis = weight * x_data + bias

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize to use gradient descent optimizer
learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# before starting, initialize the variables
init = tf.initialize_all_variables()

# launch
sess = tf.Session()
sess.run(init)


# for graph
weight_value = []
bias_value = []

# fit the line
for step in range(2001):
    sess.run(train)

    if step%20 == 0:
        weight_value.append(weight)
        bias_value.append(bias)
        print(step, sess.run(cost), sess.run(weight), sess.run(bias))

# learns best fit is weight is 1, bias is 0

