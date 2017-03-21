import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")

hypothesis = tf.add(tf.mul(weight, x), bias)

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in range(2000):
        if step % 20 == 0:
            cost_val = sess.run([train_op, cost], feed_dict={x:x_data, y:y_data})

            print(step, cost_val, sess.run(weight), sess.run(bias))

    print("=====test=======")
    print("x: 5 y: ", sess.run(hypothesis, feed_dict={x:5}))
    print("x: 2.5 y: ", sess.run(hypothesis, feed_dict={x:2.5}))