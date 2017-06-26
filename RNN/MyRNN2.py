'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
from abc import abstractmethod



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

    def nomalization(self):
        numerator = self.data - np.min(self.data, 0)
        denominator = np.max(self.data, 0) - np.min(self.data, 0)
        # noise term prevents the zero division
        self.data = numerator / (denominator + 1e-7)

    def reverse(self):
        self.data = self.data[::-1]  # reverse order (chronically ordered)

    def load(self, file_name = None, seq_length=None):
        # Open, High, Low, Volume, Close
        self.data = np.loadtxt(file_name, delimiter=',')

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
    # learning_rate = 0.01
    # iterations = 500

    hypothesis = None
    loss = None
    optimizer = None
    train = None

    X = None
    Y = None

    @abstractmethod
    def init_rnn_library(self):
        pass

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

    def learning(self, trainX=None, trainY=None, loop=None):
        tf.set_random_seed(777)  # reproducibility

        self.init_rnn_library()
        # build a LSTM network
        # self.setHypothesis()
        # self.setCostfunction()
        # self.setOptimizer()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Training step
        for i in range(loop):
            _, step_loss = self.sess.run([self.train, self.loss], feed_dict={
                self.X: trainX, self.Y: trainY})
            # print("[step: {}] loss: {}".format(i, step_loss))

        # self.prediction(testX, testY)

    def prediction(self, testX, testY):
        # test 데이터를 이용해서 예측을 해보고 표로 나타내어본다
        # RMSE
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
        # Test step
        test_predict = self.sess.run(self.hypothesis, feed_dict={self.X: testX})
        rmse_val = self.sess.run(rmse, feed_dict={
            targets: testY, predictions: test_predict})
        # print("RMSE: {}".format(rmse_val))
        # Plot predictions
        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.show()

    def setOptimizer(self, learning_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.loss)

    def setCostfunction(self):
        # cost/loss
        self.loss = tf.reduce_sum(tf.square(self.hypothesis - self.Y))  # sum of the squares

    def setHypothesis(self):
        cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.hidden_dim, state_is_tuple=True, activation=tf.tanh)
        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        self.hypothesis = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.output_dim, activation_fn=None)  # We use the last cell's output

