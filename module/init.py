import ctypes

import tensorflow as tf

a = tf.Variable(5)
b = tf.Variable(6)

c = tf.constant(11)

result = tf.mul(a,b)



init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print(id(a))
print(id(sess.run(a)))



print(tf.Print(a, [a]))
print("=====")
print(a)
print(b)
print(result)
print(c)
