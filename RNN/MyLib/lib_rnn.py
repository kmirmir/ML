import tensorflow as tf
import matplotlib
from abc import abstractmethod

matplotlib.use('TkAgg')
import matplotlib.pyplot as plot


class RNNLibrary:
    # train Parameters
    seq_length = 0
    data_dim = 0
    output_dim = 0

    hypothesis = None
    loss = None
    optimizer = None
    train = None

    X = None
    Y = None

    errors = []

    @abstractmethod
    def init_rnn_library(self):
        pass

    def setParams(self, seq_length, data_dim, output_dim):
        self.seq_length = seq_length
        self.data_dim = data_dim
        self.output_dim = output_dim

    def setPlaceholder(self, seq_length=None, data_dim=None):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

    def learning(self, trainX=None, trainY=None, loop=None):
        tf.set_random_seed(777)  # reproducibility

        self.init_rnn_library()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Training step
        for i in range(loop):
            _, step_loss = self.sess.run([self.train, self.loss], feed_dict={
                self.X: trainX, self.Y: trainY})
            # print("[step: {}] loss: {}".format(i, step_loss))
            self.errors.append(step_loss)

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
        plot.plot(testY)
        plot.plot(test_predict)
        plot.xlabel("Time Period")
        plot.ylabel("Stock Price")
        plot.show()

    def setOptimizer(self, learning_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.loss)

    def setCostfunction(self):
        self.loss = tf.reduce_sum(tf.square(self.hypothesis - self.Y))  # sum of the squares

    def setHypothesis(self, hidden_dim):
        cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        self.hypothesis = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.output_dim, activation_fn=None)  # We use the last cell's output

    def showErrors(self):
        attr = 'o-'  # 선 속성
        x_label = ''
        y_label = ''

        plot.plot(self.errors, attr)
        plot.xlabel(x_label)
        plot.ylabel(y_label)
        plot.show()
