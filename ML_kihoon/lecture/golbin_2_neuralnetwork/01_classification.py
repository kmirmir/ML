# -*- coding: utf-8 -*-
# 털과 날개 유무에 따라, 포유류인지 조류인지 분류하는 신경망 모델

import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

# [기타, 포유류, 조류]
y_data = np.array(
    [
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1] ]
)

##################3
### 신경망 모델 구성
###################

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

weight1 = tf.Variable(tf.random_uniform([2,10], -1.0, 1.0))
weight2 = tf.Variable(tf.random_uniform([10, 3], -1.0, 1.0))

bias1 = tf.Variable(tf.zeros([10]))
bias2 = tf.Variable(tf.zeros([3]))

layer = tf.add(tf.matmul(x, weight1), bias1)
layer = tf.nn.relu(layer)

model = tf.add(tf.matmul(layer, weight2), bias2)
model = tf.nn.softmax(model)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(model)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

################
#### 신경망 모델 학습
################

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train_op, feed_dict={x: x_data, y: y_data})

    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={x:x_data, y:y_data}))


################
#### 결과 확인
#### 0: 기타, 1: 포유류, 2: 조류
################
print("\n")

prediction = tf.argmax(model, 1)
target = tf.argmax(y, 1)

print("예측 값: ", sess.run(prediction, feed_dict={x:x_data}))
print("실제 값: ", sess.run(target, feed_dict={y:y_data}))

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print("정확도: %.2f" % sess.run(accuracy * 100, feed_dict={x:x_data, y:y_data}))

########
## 내 실험 값
#######
print("예측 값: ", sess.run(prediction, feed_dict={x:[[1,0],[1,0],[0,0],[0,0],[1,1],[0,1]]}))

