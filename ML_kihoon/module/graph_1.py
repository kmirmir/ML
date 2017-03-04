import tensorflow as tf

a = tf.Variable(5)

print(a)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print(sess.run(a))