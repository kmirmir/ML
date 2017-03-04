import tensorflow as tf
from matplotlib import pyplot as plt

input_x_data = [1,2,3]
input_y_data = [1,2,3]

# tf.Variable -->> tf.placeholder changed!!
# this problem occur the error, tensorflow shape
# weight = tf.placeholder("float")
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = weight * input_x_data + bias
cost = tf.reduce_mean(tf.square(hypothesis - input_y_data))

optimize = tf.train.GradientDescentOptimizer(0.1)
train = optimize.minimize(cost)




init = tf.initialize_all_variables()



# for graphs
W_val = []
cost_val = []


sess = tf.Session()
sess.run(init)

# if remove the feed_dict of weight , the graph is linear

for i in range(-30, 50):
    print(i * -0.1, sess.run(cost))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost))

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()

plt.plot(input_x_data, input_y_data, 'ro', label = "data")
plt.plot(input_x_data, sess.run(weight)*input_x_data+sess.run(bias), 'ro', lable = "hypothesis")
plt.show()
