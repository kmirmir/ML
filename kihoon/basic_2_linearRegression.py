import tensorflow as tf

# training data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# if H(x) = Weight * x_data + bias
# we know idea thing is H(x) = 1*x + b
# Is tensorflow know the how to learning?
# we give random variable -1.0 ~ 1.0 in Weight and bias
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Hypothesis = Weight * x_data + bias
hypothesis = weight*x_data + bias

# cost function(weight, bias)
# only set the operation, not running
costFunction = tf.reduce_mean(tf.square(hypothesis-y_data))

# Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(costFunction)


# before session run,
# starting initialize the variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)

    if step%20 == 0:
        print(step, sess.run(costFunction), sess.run(weight), sess.run(bias))