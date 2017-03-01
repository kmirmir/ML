import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class machin:

    def set_data(self, input_x, input_y):
        self.x_data = input_x
        self.y_data = input_y
        print("Setting the data is finish")


    def learning(self, learningRate):
        print("Starting learning to data")
        # try to find values for w and b that computer is compute y_data = weight * x_data + b
        self.weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        self.bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

        # my hypothesis
        self.hypothesis = self.weight * self.x_data + self.bias

        # cost function
        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.y_data))

        # minimize to use gradient descent optimizer
        self.learning_rate = tf.Variable(learningRate)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        # before starting, initialize the variables
        init = tf.initialize_all_variables()

        # launch
        with tf.Session() as sess:
            sess.run(init)

            # fit the line
            for step in range(2001):
                sess.run(self.train)

                # sess.run log for display to know changing the wehight and bias, cost
                if step%20 == 0:
                    print(step, sess.run(self.cost), sess.run(self.weight), sess.run(self.bias))



            # graph for
            plt.plot(self.x_data, self.y_data, 'ro', label = 'data')
            plt.plot(self.x_data, sess.run(self.weight)*self.x_data+sess.run(self.bias), label = 'Hypothesis')
            plt.legend()
            plt.show()

    def set_test_data(self, input_x):

        with tf.Session() as sess:
            print(sess.run(self.hypothesis, feed_ditc={self.x_data : input_x}))

        # plt.plot(input_x, input_y, 'ro', label='data')
        # plt.plot(input_x, sess.run(self.weight) * input_x + sess.run
        # (self.bias), label='Hypothesis')
        # plt.legend()
        # plt.show()

if __name__ == '__main__':
    kihoon = machin()
    kihoon.set_data([1,2,3], [1,2,3])
    kihoon.learning(0.1)

    kihoon.set_test_data([9,10])



