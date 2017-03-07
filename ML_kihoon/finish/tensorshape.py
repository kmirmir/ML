import tensorflow as tf

a = tf.placeholder(tf.float32, [None, None])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


print(a.value())