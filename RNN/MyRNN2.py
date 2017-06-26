'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
from abc import abstractmethod

tf.set_random_seed(777)  # reproducibility

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Database:
    data = []
    trainX = []
    trainY = []
    testX = []
    testY = []

    @abstractmethod
    def init_dataset(self):
        pass

    def nomalization(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return numerator / (denominator + 1e-7)

    def load(self, file_name = None, seq_length=None):
        # Open, High, Low, Volume, Close
        self.data = np.loadtxt(file_name, delimiter=',')
        self.data = self.data[::-1]  # reverse order (chronically ordered)
        # xy = self.nomalization(xy)
        self.init_dataset()

        x = self.data
        y = self.data[:, [-1]]  # Close as label

        # build a dataset
        dataX = []
        dataY = []

        for i in range(0, len(y) - seq_length):
            _x = x[i:i + seq_length]
            _y = y[i + seq_length]  # Next close price
            # print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)

        # train/test split
        train_size = int(len(dataY) * 0.7)
        test_size = len(dataY) - train_size
        self.trainX, self.testX = np.array(dataX[0:train_size]), np.array(
            dataX[train_size:len(dataX)])
        self.trainY, self.testY = np.array(dataY[0:train_size]), np.array(
            dataY[train_size:len(dataY)])


class RNNLibrary:
    # train Parameters
    seq_length = 0
    data_dim = 0
    output_dim = 0

    hidden_dim = 10
    learning_rate = 0.01
    iterations = 500

    X = None
    Y = None

    def setParams(self, seq_length, data_dim, output_dim):
        self.seq_length = seq_length
        self.data_dim = data_dim
        self.output_dim = output_dim
        # seq_length = 7
        # data_dim = 5
        # output_dim = 1

    def setPlaceholder(self, seq_length=None, data_dim=None):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

    def run(self, trainX=None, trainY=None, testX=None, testY=None):
        # build a LSTM network
        cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.hidden_dim, state_is_tuple=True, activation=tf.tanh)
        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.output_dim, activation_fn=None)  # We use the last cell's output

        # cost/loss
        loss = tf.reduce_sum(tf.square(Y_pred - self.Y))  # sum of the squares
        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        # RMSE
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(self.iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                self.X: trainX, self.Y: trainY})
            # print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={self.X: testX})
        rmse_val = sess.run(rmse, feed_dict={
                        targets: testY, predictions: test_predict})
        # print("RMSE: {}".format(rmse_val))

        # Plot predictions
        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.show()

