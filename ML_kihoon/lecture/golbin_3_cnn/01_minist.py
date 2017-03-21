import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.examples.tutorials.mnist import mnist
# from tensorflow.models.image import mnist
# from tensorflow.examples.tutorials import mnist

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weight1 = tf.Variable(tf.random_normal([784, 256], stddev = 0.1))
weight2 = tf.Variable(tf.random_normal([256, 256], stddev = 0.1))
weight3 = tf.Variable(tf.random_normal([256, 10], stddev = 0.1))

layer1 = tf.nn.relu(tf.matmul(x, weight1))
layer2 = tf.nn.relu(tf.matmul(layer1, weight2))
model = tf.matmul(layer2, weight3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
        # 지정한 크기만큼 학습할 데이터를 가져옵니다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        total_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

    print( 'Epoch:', '%04d' % (epoch + 1),\
            'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print ('최적화 완료!')

check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print ('정확도:', sess.run(accuracy,
                            feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels}))