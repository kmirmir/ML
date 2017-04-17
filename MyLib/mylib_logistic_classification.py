import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class logistic:

    W_val = []
    cost_val = []



    def __init__(self, x_data, y_data):

        self.train_x = x_data
        self.train_y = y_data

        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)

        self.W = tf.Variable(tf.random_uniform([1, len(self.train_x)], -1.0, 1.0))

        h = tf.matmul(self.W, self.X)
        self.hypothesis = tf.div(1., 1. + tf.exp(-h))

        self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))


        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def running(self, show_or_hidden, loop_size, print_point):
        # a = tf.Variable(0.1)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train = optimizer.minimize(self.cost)

        self.sess.run(self.init)

        if show_or_hidden == 1:
            for step in range(loop_size):
                self.sess.run(self.train, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                if step % print_point == 0:
                    print ( step, "단계 : ",
                           "에러 : ", self.sess.run(self.cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}),
                           "기울기(Weight) : ", self.sess.run(self.W))


                    # for graphs
                    self.W_val.append(self.W_val.append(self.sess.run(self.W)))
                    self.cost_val.append(self.sess.run(self.cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}))


        elif show_or_hidden == 2:
            for step in range(loop_size):
                self.sess.run(self.train, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                if step % print_point == 0:
                    self.sess.run(self.cost, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                    # for graphs
                    self.W_val.append(self.W_val.append(self.sess.run(self.W)))
                    self.cost_val.append(self.sess.run(self.cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}))

        else:
            pass


    def show_cost(self):
        plt.plot(self.cost_val, 'o-', label='cost')
        plt.legend()
        plt.show()



    def test_running(self, test_x):
        print ('-------------')

        print (self.sess.run(self.hypothesis, feed_dict={self.X: test_x}) > 0.5)
        # print (self.sess.run(self.hypothesis, feed_dict={self.X: [[1], [5], [5]]}) > 0.5)
        # print (self.sess.run(self.hypothesis, feed_dict={self.X: [[1, 1], [4, 3], [3, 5]]}) > 0.5)

if __name__ == '__main__':
    train_data = np.loadtxt('train.txt', unpack=True, dtype='float32')

    x_data = train_data[0:-1]
    y_data = train_data[-1]

    test_x = [[1], [2], [2]]

    kihoon = logistic(x_data, y_data)
    kihoon.set_learning_rate(0.1)
    kihoon.running(1,2001,20)
    kihoon.test_running(test_x)


    # kihoon.show_cost()