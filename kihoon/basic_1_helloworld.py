import tensorflow as tf

hello = tf.constant('Hello TensorFlow!')
print(hello)

sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)

print (a)
print (sess.run(a+b))

# Close the Session when we're done.
sess.close()