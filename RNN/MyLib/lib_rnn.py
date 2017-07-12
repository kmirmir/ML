import sys
import tensorflow as tf
import matplotlib
from abc import abstractmethod

matplotlib.use('TkAgg')
import matplotlib.pyplot as plot


class RNNLibrary:
    # train Parameters
    seq_length = 0
    input_dim = 0
    output_dim = 0

    hypothesis = None
    cost = None
    optimizer = None
    train = None

    X = None
    Y = None

    errors = []
    epoch_cost = []

    @abstractmethod
    def init_rnn_library(self):
        pass

    def setParams(self, seq_length, input_dim, output_dim):
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def setPlaceholder(self, seq_length=None, input_dim=None):
        self.X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

    def learning(self, trainX=None, trainY=None, loop=None, total_epoch = 1, check_step=0):
        tf.set_random_seed(777)  # reproducibility

        self.init_rnn_library()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(total_epoch):
            total_cost = 0

            for i in range(loop):
                _, step_loss = self.sess.run([self.train, self.cost], feed_dict={self.X: trainX, self.Y: trainY})
                print("[step: {}] loss: {}".format(i, step_loss))
                self.errors.append(step_loss)
                total_cost += step_loss
                self.epoch_cost.append(total_cost)

                if i % check_step == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        print('\nDone!\n')

    def setHypothesis(self, hidden_dim, layer=1, isDropout=False, dropout_value=1):
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh, reuse=tf.get_variable_scope().reuse
            )
            if isDropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_value)

            return cell

        multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(layer)], state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(multi_cell, self.X, dtype=tf.float32)

        self.hypothesis = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim, activation_fn=None)

    def prediction(self, testX, testY):
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

        test_predict = self.sess.run(self.hypothesis, feed_dict={self.X: testX})
        rmse_val = self.sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})

        print("RMSE: {}".format(rmse_val))
        print("RMSE: {:.2%}".format(rmse_val))

        plot.plot(testY)
        plot.plot(test_predict)
        plot.xlabel("Test Size (blue is TestY, orange is Predict")
        plot.ylabel("Invertor Output")
        plot.show()

    def setCostfunction(self):
        self.cost = tf.reduce_sum(tf.square(self.hypothesis - self.Y))  # sum of the squares

    def setOptimizer(self, learning_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.cost)

    def showErrors(self):
        attr = 'o-'  # 선 속성
        x_label = ''
        y_label = ''

        plot.plot(self.errors, attr)
        plot.xlabel(x_label)
        plot.ylabel(y_label)
        plot.show()