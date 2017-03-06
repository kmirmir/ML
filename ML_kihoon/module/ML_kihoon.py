# import librarys
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

# this source is one-variable linear regression
# i do modulationa in the

class machineLearning:

    def set_data(self, x_data, y_data):
        self.input_x_data = [1, 2, 3]
        self.input_y_data = [2, 4, 6]


    def set_data_from_file(self, filepath):
        file_data = np.loadtxt(filepath, unpack=True, dtype='float32')
        self.input_x_data = file_data[0:-1]
        self.input_y_data = file_data[-1]


    def learning(self, input_learning_rate):

        # for graphs
        self.W_val = []
        self.cost_val = []

        self.x_data = tf.placeholder(tf.float32)
        self.y_data = tf.placeholder(tf.float32)

        self.weight = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0))
        self.bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

        self.hypothesis = self.weight * self.x_data + self.bias

        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.y_data))

        self.learning_rate = tf.Variable(input_learning_rate)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        # before starting, initialize the variables
        init = tf.initialize_all_variables()

        # launch
        self.sess = tf.Session()
        self.sess.run(init)

        # fit the line
        for step in range(2001):
            self.sess.run(self.train, feed_dict={self.x_data: self.input_x_data, self.y_data: self.input_y_data})

            self.W_val.append(self.sess.run(self.weight))
            self.cost_val.append(self.sess.run(self.cost, feed_dict={self.x_data: self.input_x_data, self.y_data: self.input_y_data}))
            if step % 20 == 0:
                print(step, self.sess.run(self.cost, feed_dict={self.x_data: self.input_x_data, self.y_data: self.input_y_data}), self.sess.run(self.weight), self.sess.run(self.bias))


    def set_test_data(self, input_x):
        self.test_input_x_data = input_x
        self.test_data = self.sess.run(self.hypothesis, feed_dict={self.x_data : self.test_input_x_data})
        print(self.test_data)

    def session_close(self):
        self.sess.close()

    def show_set_data_to_graph(self):
        # graph for
        plt.plot(self.input_x_data, self.input_y_data, 'ro', label='data')
        plt.plot(self.input_x_data, self.sess.run(self.weight) * self.input_x_data + self.sess.run(self.bias), label='Hypothesis')
        plt.legend()
        plt.show()

    def show_test_data_to_graph(self):
        # graph for
        plt.plot(self.test_input_x_data, self.test_data, 'ro', label='data')
        plt.plot(self.test_input_x_data, self.sess.run(self.weight) * self.test_input_x_data + self.sess.run(self.bias), label='Hypothesis')
        plt.legend()
        plt.show()


    def show_cost_value(self):
        plt.plot(self.W_val, self.cost_val, 'ro')
        plt.ylabel('cost')
        plt.xlabel('W')
        plt.show()


if __name__ == '__main__':
    # setting
    # set_data is [1,2,3], [1,2,3] --> learning rate is 0.1 right rate
    kihoon = machineLearning()
    kihoon.set_data([1, 2, 3], [2, 4, 6])
    kihoon.learning(0.1)
    # kihoon.show_cost_value()
    # kihoon.show_set_data_to_graph()

    kihoon.set_test_data([7., 9.])
    # kihoon.show_test_data_to_graph()

    kihoon.session_close()