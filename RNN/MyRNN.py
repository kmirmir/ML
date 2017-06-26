import tensorflow as tf
import numpy as np
import matplotlib

tf.set_random_seed(777)  # reproducibility

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RNN:
    # train Parameters
    seq_length = 0
    data_dim = 0
    output_dim = 0

    hidden_dim = 10
    # build a dataset

    dataX = []
    dataY = []

    trainX = []
    trainY = []
    testX = []
    testY = []

    x = None
    y = None

    X = None
    Y = None

    def setParams(self, data_dim, seq_length, output_dim):
        self.data_dim = data_dim
        self.seq_length = seq_length
        self.output_dim = output_dim

    def MinMaxScaler(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return numerator / (denominator + 1e-7)

    def loadData(self, fileName):
        # Open, High, Low, Volume, Close
        xy = np.loadtxt(fileName, delimiter=',')
        xy = xy[::-1]  # reverse order (chronically ordered)
        xy = self.MinMaxScaler(xy)

        self.x = xy
        self.y = xy[:, [-1]]  # Close as label

    def buildData(self):
        for i in range(0, len(self.y) - self.seq_length):
            _x = self.x[i:i + self.seq_length]
            _y = self.y[i + self.seq_length]  # Next close price
            print(_x, "->", _y)
            self.dataX.append(_x)
            self.dataY.append(_y)

    def buildTrainDataPercentage(self, percentage):
        percentage = percentage / 100
        # train/test split
        train_size = int(len(self.dataY) * percentage)
        test_size = len(self.dataY) - train_size
        self.trainX, self.testX = np.array(self.dataX[0:train_size]), np.array(
            self.dataX[train_size:len(self.dataX)])
        self.trainY, self.testY = np.array(self.dataY[0:train_size]), np.array(
            self.dataY[train_size:len(self.dataY)])

    def setPlaceholder(self, seq_length, data_dim):
        seq_length = self.seq_length
        data_dim = self.data_dim

        # input place holders
        self.X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

    def run(self, learning_rate, iterations):
        # build a LSTM network
        cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.hidden_dim, state_is_tuple=True, activation=tf.tanh)
        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.output_dim, activation_fn=None)  # We use the last cell's output

        # cost/loss
        loss = tf.reduce_sum(tf.square(Y_pred - self.Y))  # sum of the squares
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        # RMSE
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                self.X: self.trainX, self.Y: self.trainY})
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        self.test_predict = sess.run(Y_pred, feed_dict={self.X: self.testX})
        # rmse_val = sess.run(rmse, feed_dict={
        #     targets: self.testY, predictions: self.test_predict})
        # print("RMSE: {}".format(rmse_val))


    def plt(self, testY=None, test_predict=None):
        # Plot predictions
        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.show()


if __name__ == '__main__':
    rnn = RNN()
    rnn.setParams(5,7,1)
    rnn.setPlaceholder(seq_length=rnn.seq_length, data_dim=rnn.data_dim)
    rnn.loadData('data-02-stock_daily.csv')
    rnn.buildData()
    rnn.buildTrainDataPercentage(70)
    rnn.run(0.01, 500)
    rnn.plt(rnn.testY, rnn.test_predict)