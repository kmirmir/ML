import tensorflow as tf

hello = tf.constant('Hello Tensorflow!')
print(hello)

sess = tf.Session()
print(sess.run(hello))

a = tf.placeholder("float")
b = tf.placeholder("float")

result = sess.run(tf.add(a,b), feed_dict={a:10., b:20.})
print(result)