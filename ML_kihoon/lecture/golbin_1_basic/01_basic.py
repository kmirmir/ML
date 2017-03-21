import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')
hello2 = tf.Variable('hello')

a = tf.constant(10)
b = tf.constant(32)
c = a+b

x = tf.placeholder("float", [None, 3])

weight = tf.Variable(tf.random_normal([3,2]), name='Weights')
bias = tf.Variable(tf.random_normal([2,1]), name='Bias')

x_data = [[1,2,3], [4,5,6]]

expr = tf.matmul(x, weight) + bias

sess = tf.Session()
init = tf.initialize_all_variables()

sess.run(init)


print("------constants--------")
print(sess.run(hello))
print(sess.run(hello2))

print("-------x_data--------")
print(sess.run(c))

print("-------weight--------")
print(sess.run(weight))

print("--------bias---------")
print(sess.run(bias))

print("------expr-------")
print(sess.run(expr, feed_dict={x: x_data}))

sess.close()