import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class multiLinearRegression:

    weightVal = []
    costVal = []

    def __init__(self):
        self.xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
        self.x_data = self.xy[0:-1]
        self.y_data = self.xy[-1]

        print(self.x_data)
        print(self.y_data)

        self.weight = tf.Variable(tf.random_uniform([1, len(self.x_data)], -1.0, 1.0))
        self.hypothesis = tf.matmul(self.weight, self.x_data)
        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.y_data))

        self.learning_rate = tf.placeholder('float')
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def learning(self, learning_rate):

        for step in range(2001):
            self.sess.run(self.train, feed_dict={self.learning_rate:learning_rate})
            self.costVal.append(self.sess.run(self.cost))
            self.weightVal.append(self.sess.run(self.weight))

            if step % 20 == 0:
                print(step, self.sess.run(self.cost), self.sess.run(self.weight))

if __name__ == '__main__':
    kihoon = multiLinearRegression()
    kihoon.learning(0.1)
