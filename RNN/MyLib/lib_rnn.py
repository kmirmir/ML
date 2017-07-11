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

    @abstractmethod
    def init_rnn_library(self):
        pass

    def setParams(self, seq_length, input_dim, output_dim):
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def setPlaceholder(self, seq_length=None, input_dim=None):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

    def learning(self, trainX=None, trainY=None, loop=None, check_step=0):
        tf.set_random_seed(777)  # reproducibility

        # init_rnn_library()를 이용해서 rnn model을 setting할 수 있다
        # setting된 걸 불러와서 사용
        self.init_rnn_library()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Training step
        for i in range(loop):
            _, step_loss = self.sess.run([self.train, self.cost], feed_dict={self.X: trainX, self.Y: trainY})
            print("[step: {}] loss: {}".format(i, step_loss))
            self.errors.append(step_loss)

            if i % check_step == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

        print('\nDone!\n')

    def prediction(self, testX, testY):
        # test 데이터를 이용해서 예측을 해보고 표로 나타내어본다
        # RMSE
        # RMSE가 작을수록 좋은거!!!
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

        # Test step
        # 테스트 x 데이터를 놓고 테스트 x에 대한 결과 값을 test_predict에 저장한다
        # 24*7*4*0.3 = 테스트 사이즈
        # 24*7*4*0.7 = 트레인 사이즈
        test_predict = self.sess.run(self.hypothesis, feed_dict={self.X: testX})
        rmse_val = self.sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})


        print("RMSE: {}".format(rmse_val))
        print("RMSE: {:.2%}".format(rmse_val))
        # print("RMSE: {:.2%}".format(rmse_val))


        # Plot predictions
        # 주황색이 testY, 파랑색이 predict된 값들
        # plot.plot(testY, c="b", lw=5)
        plot.plot(testY)
        plot.plot(test_predict)
        # plot.xlim(0, 100)
        # plot.ylim(-1, 9)
        plot.xlabel("Test Size (blue is TestY, orange is Predict")
        plot.ylabel("Invertor Output")
        plot.show()

    def setHypothesis(self, hidden_dim, layer = 1, isDropout = False, dropout_value = 1):
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh, reuse=tf.get_variable_scope().reuse
            )
            if isDropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_value)

            return cell
        # 됨요~ reuse 써야 되고이 그 함수를 불러오면 잘 되긴하네 근데 loss는 너무 높다
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(layer)], state_is_tuple=True)

        # drop out이나 hidden dimension을 조절하고 싶다면 아래처럼 그냥 하나로 통일하고 돌리고 싶다면 위처럼
        # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)
        cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # cell2 = tf.con®rnn_cell.DropoutWrapper(cell2, output_keep_prob=0.5)
        # cell3 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # cell4 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # cell5 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        # multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell2,cell3,cell4,cell5])

        # 단일 셀만 돌릴거면 multi_cell --> cell로 바꿔주기만 하면 된다.
        outputs, _states = tf.nn.dynamic_rnn(multi_cell, self.X, dtype=tf.float32)

        # We use the last cell's output
        self.hypothesis = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_dim, activation_fn=None)

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