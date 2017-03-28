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

        feed_dict = {self.X: x_data, self.Y: y_data}

        for step in range(2001):
            self.sess.run(optimizer, feed_dict=feed_dict)
            if step % 200 == 0:
                print(step, self.sess.run(cost, feed_dict=feed_dict))

    def predict(self):
        print('--------------')

        # Testing & One-hot encoding
        a = self.sess.run(self.hypothesis, feed_dict={self.X: [[1, 11, 7, 9]]})
        print(a, self.sess.run(tf.arg_max(a, 1)))
        print('--------------')
        b = self.sess.run(self.hypothesis, feed_dict={self.X: [[1, 3, 4, 3]]})
        print(b, self.sess.run(tf.arg_max(b, 1)))
        print('--------------')

        c = self.sess.run(self.hypothesis, feed_dict={self.X: [[1, 1, 0, 1]]})
        print(c, self.sess.run(tf.arg_max(c, 1)))

        print('--------------')
        all = self.sess.run(self.hypothesis, feed_dict={self.X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
        print(all, self.sess.run(tf.arg_max(all, 1)))


if __name__ == '__main__':
    x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
              [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
    y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
              [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

    print(len(x_data[0]))
    print(len(y_data[0]))

    kihoon = softmax(x_data, y_data)
    kihoon.run()
    kihoon.predict()