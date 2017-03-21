import tensorflow as tf
import numpy as np

f = open ('./data.csv', encoding='utf-8')
header = f.readline() # .split()[1].split(',')
data = np.loadtxt(f, delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

weight1 = tf.Variable(tf.random_uniform([2,10], -1.0 ,1.0))
weight2 = tf.Variable(tf.random_uniform([10,20], -1.0, 1.0))
weight3 = tf.Variable(tf.random_uniform([20,3], -1.0, 1.0))

layer1 = tf.nn.relu(tf.matmul(x, weight1))
layer2 = tf.nn.relu(tf.matmul(layer1, weight2))

model = tf.matmul(layer2, weight3)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(model, y)
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train_op, feed_dict={x: x_data, y: y_data})

    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}))

prediction = tf.argmax(model, 1)
target = tf.argmax(y, 1)

print("예측값 : ", sess.run(prediction, feed_dict={x:x_data}))
print("실제값 : ", sess.run(target, feed_dict={y:y_data}))

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print("정확도 : %.2f"% sess.run(accuracy*100, feed_dict={x:x_data, y:y_data}))