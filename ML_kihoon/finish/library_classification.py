import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

"""
classification
1. logistic classification
2. softmax classification
"""
class classification:
    def __init__(self, feature):
        self.W_val = []
        self.cost_val = []
        self.feature = feature

        self.learning_x_data = tf.placeholder(tf.float32)
        self.learning_y_data = tf.placeholder(tf.float32)

        if self.feature == "logistic":
            self.weight = tf.Variable(tf.random_uniform([1,3],-1.0,1.0))
            self.h = tf.matmul(self.weight, self.learning_x_data)
            self.hypothesis = tf.div(1., 1. + tf.exp(-self.h))
            self.cost = -tf.reduce_mean(self.learning_y_data * tf.log(self.hypothesis) + (1 - self.learning_y_data) * tf.log(1 - self.hypothesis))


        elif self.feature == "softmax":
            pass

        else:
            pass

        self.learning_rate = tf.placeholder('float')
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def set_data_default(self):
        if self.feature == "logistic":
            self.input_x_data = []
            self.input_y_data = []

        elif self.feature == "softmax":
            pass

        else:
            pass

    def set_data_loadFile(self):
        xy = np.loadtxt('training_logistic.txt', unpack=True, dtype='float32')
        self.input_x_data = xy[0:-1]
        self.input_y_data = xy[-1]
        print(self.input_x_data)
        print(self.input_y_data)

    def set_test_data_default(self):
        if self.feature == "logistic":
            pass

        elif self.feature == "softmax":
            pass

        else:
            pass

    def learning(self, learning_rate):
        if self.feature == "logistic":
            for step in range(2001):
                self.sess.run(self.train,
                              feed_dict={self.learning_x_data: self.input_x_data,
                                         self.learning_y_data: self.input_y_data,
                                         self.learning_rate: learning_rate})

                self.W_val.append(self.sess.run(self.weight))
                self.cost_val.append(self.sess.run(self.cost,
                                                   feed_dict={self.learning_x_data: self.input_x_data,
                                                              self.learning_y_data: self.input_y_data}))

                if step % 20 == 0:
                    print("step:", step,
                          " weight:", self.sess.run(self.weight),
                          " cost:", self.sess.run(self.cost,
                                                  feed_dict={self.learning_x_data: self.input_x_data,
                                                             self.learning_y_data: self.input_y_data}))
        elif self.feature == "softmax":
            pass

        else:
            pass

    def show_input_data(self):
        if self.feature == "logistic":
            plt.plot(self.input_x_data, self.input_y_data, 'ro')
            plt.legend()
            plt.show()

        elif self.feature == "softmax":
            pass

        else:
            pass

    def show_test_data(self):
        if self.feature == "logistic":
            print('---------------------------')

            print(self.sess.run(self.hypothesis, feed_dict={self.learning_x_data: [[1], [2], [2]]}) > 0.5)
            print(self.sess.run(self.hypothesis, feed_dict={self.learning_x_data: [[1], [5], [5]]}) > 0.5)
            print(self.sess.run(self.hypothesis, feed_dict={self.learning_x_data: [[1, 1], [4, 3], [3, 5]]}) > 0.5)


        elif self.feature == "softmax":
            pass

        else:
            pass

    def show_cost_data(self):
        if self.feature == "logistic":
            plt.plot(self.W_val, self.cost_val, 'ro')
            plt.show()

        elif self.feature == "softmax":
            pass

        else:
            pass

if __name__ == '__main__':
    kihoon = classification("logistic")
    # kihoon.set_data_default()
    kihoon.set_data_loadFile()
    kihoon.learning(0.1)

    # 에러 난다. 2017/03/07 am 01:09
    # kihoon.show_input_data()
    # kihoon.show_cost_data()

    kihoon.show_test_data()