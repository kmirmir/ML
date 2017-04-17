# Lab 6 Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

class softmax:

    x_data = []
    y_data = []

    W = None
    X = None
    Y = None
    sess = None
    init = None

    W_val = []
    cost_val = []

    def __init__(self, x_data, y_data):
        self.X = tf.placeholder("float", [None, len(x_data[0])])
        self.Y = tf.placeholder("float", [None, len(y_data[0])])
        nb_classes = len(y_data[0])

        self.W = tf.Variable(tf.random_normal([len(x_data[0]), nb_classes]), name='weight')
        self.b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

        self.feed_dict = {self.X: x_data, self.Y: y_data}

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()

    def run(self):
        # tf.nn.softmax computes softmax activations
        # softmax = exp(logits) / reduce_sum(exp(logits), dim)
        self.hypothesis = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

        # Cross entropy cost/loss
        # cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
        cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), reduction_indices=1))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

        # Launch graph
        self.sess.run(self.init)


        for step in range(2001):
            self.sess.run(optimizer, feed_dict=self.feed_dict)
            if step % 200 == 0:
                print(step, self.sess.run(cost, feed_dict=self.feed_dict))

    def predict(self, feed_dict):
        feeding = {self.X: feed_dict}
        print('--------------')
        all = self.sess.run(self.hypothesis, feed_dict=feeding)
        print(all, self.sess.run(tf.arg_max(all, 1)))

    def prediction(self):
        prediction = tf.argmax(self.hypothesis, 1)
        target = tf.argmax(self.Y, 1)
        print('예측값:', self.sess.run(prediction, feed_dict={self.X: x_data}))
        print('실제값:', self.sess.run(target, feed_dict={self.Y: y_data}))

        check_prediction = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
        print('정확도: %.2f' % self.sess.run(accuracy * 100, feed_dict=self.feed_dict))


'''
[[1, 11, 7, 9]]
[[1, 3, 4, 3]]
[[1, 1, 0, 1]]
[[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]
'''

'''
데이터 셋 : 뭐먹을까?
one-hot 데이터 매운 정도 따듯한 정도  밥 있음 없음
'''
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

kihoon = softmax(x_data, y_data)
kihoon.run()
kihoon.predict([[1, 11, 7, 9]])
kihoon.prediction()



'''
직방 데이터 셋
'''