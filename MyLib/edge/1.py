import tensorflow as tf

a = tf.constant(5, name='input_a')
b = tf.constant(3, name='input_b')
c = tf.mul(a,b, name='mul_c')
d = tf.add(a,b, name='add_d')
e = tf.add(c,d, name='add_e')

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

output = sess.run(e)

# writer = tf.train.SummaryWriter('./edge', sess.graph)
sess.close()