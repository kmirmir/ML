# import tensorflow, matplotlib.pyplot, numpy
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


"""
linear regression 클래스
1. one variable linear regression
2. multi variable linear regression
"""

class linearRegression:

    """
        init에서는 텐서플로우에서 구성하는 구성문들을 넣어 진행한다.
        __init__(self, one_or_multi) 를 사용하면 one variable인지 multi variable인지 명시할 수 있다.
    """

    def __init__(self, feature):
        self.W_val = []
        self.cost_val = []
        self.feature = feature

        self.learning_x_data = tf.placeholder(tf.float32)
        self.learning_y_data = tf.placeholder(tf.float32)


        if self.feature == "one_variable":
            self.weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            self.bias = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

            self.hypothesis = self.weight * self.learning_x_data + self.bias

        elif self.feature == "multi_variable":
            # 주석 해제 시, 에러 남. 2017/03/06 pm 11:20
            # self.len_x_data = tf.placeholder(tf.int32)
            self.weight = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
            # self.weight = tf.Variable(tf.random_uniform([1,self.len_x_data.eval(self.sess)], -1.0, 1.0))
            # here is why no bias?

            # 230줄 set_data_default 주석을 풀었을 때 에러가 난다
            # 왜인지 보니 https://www.tensorflow.org/api_docs/python/tf/matmul
            # 행렬의 곱에서 1,3 이면 3, 1이 되어야 하는데 그렇게 안되어있는 것 같음
            # 에러 난 시각 . 2017/03/07 pm 11:41
            self.hypothesis = tf.matmul(self.weight, self.learning_x_data)

        else:
            pass


        self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.learning_y_data))
        self.learning_rate = tf.placeholder('float')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)


    """
        데이터 세팅과 관련된 함수들
        set_data_input은 사용자으로부터 데이터를 직접 입력받는다.
        parameter is only self : 미리 입력된 데이터로 학습을 진행한다
        parameter is inputed for user : 사용자가 학습 데이터를 입력한다.

        set_data_loadFile은 사용자가 사용할 데이터 파일명을 입력받아 로드해서 입력한다.
        parameter is only self : 미리 연결된 파일로 학습 데이터를 입력받아서 학습을 진행한다
        parameter is inputed for user : 로딩할 파일 이름을 넣어서 학습을 진행한다

        set_data_random은 사용자가 넣지 않을 때 랜덤함수로 만든 데이터들을 입력한다.
    """

    def set_data_default(self):
        if self.feature == "one_variable":
            self.input_x_data = [1.,2.,3.]
            self.input_y_data = [2.,4.,6.]

        elif self.feature == "multi_variable":
            self.input_x_data = [[1.,1.,1.,1.],
                                 [1.,0.,3.,5.],
                                 [0.,2.,0.,0.]]
            self.input_y_data = [1.,2.,3.,5.]

        else:
            pass

    def set_data_input(self, input_x_data, input_y_data):
        self.input_x_data = input_x_data
        self.input_y_data = input_y_data

    # 파일에서 데이터 가지고와서 돌려도 에러 뜸. /2017/3/6 pm10;00
    # one and multi 둘 다 에러가 뜬다.
    # 아마 내가 생각하기로는 learning에서 weight의 배열을 만들 때, 텐서 값으로 들어가서 그런듯 싶다.
    def set_data_loadFile_default(self):
        if self.feature == "one_variable":
            file_data = np.loadtxt('training_one_variable.txt', unpack=True, dtype='float32')
            self.input_x_data = file_data[0:-1]
            self.input_y_data = file_data[-1]

        elif self.feature == "multi_variable":
            file_data = np.loadtxt('training_multi_variable.txt', unpack=True, dtype='float32')
            self.input_x_data = file_data[0:-1]
            self.input_y_data = file_data[-1]
            self.len_input_x_data = len(self.input_x_data)

        else:
            text_error = "no feature"
            print(text_error)

    def set_data_loadFile_with_path(self, file_path):
        pass

    def set_data_random(self):
        pass

    def set_test_data_default(self):
        if self.feature == "one_variable":
            self.test_input_x_data = [7., 9., 11.]
            self.test_data = self.sess.run(self.hypothesis,
                                           feed_dict={self.learning_x_data: self.test_input_x_data})
            print("predict : ", self.test_data)

        elif self.feature == "multi_variable":
            self.test_input_x_data = [1.,0.,3.,5.]
            self.test_input_x_data = self.sess.run(self.hypothesis,
                                                   feed_dict={self.learning_x_data: self.test_input_x_data})
            print("predict : ", self.test_data)

    def set_test_data_input(self):
        pass


    """
        데이터를 입력받은 상태에서 학습을 진행한다.
        학습 함수는 두 가지로 나눌 수 있다 인자의 유무로 나눈다.
        learning(self) 미리 정해놓은 learning rate로 학습을 진행한다.
        learning(self, learning_rate) learning_rate 학습율을 직접 입력받을 수 있게 한다.
    """


    def learning(self, learning_rate):
        for step in range(2001):
            if self.feature == "one_variable":
                self.sess.run(self.train,
                              feed_dict={self.learning_x_data: self.input_x_data,
                                         self.learning_y_data: self.input_y_data,
                                         self.learning_rate: learning_rate})
            elif self.feature == "multi_variable":
                self.sess.run(self.train,
                              feed_dict={self.learning_x_data: self.input_x_data,
                                         self.learning_y_data: self.input_y_data,
                                         self.learning_rate: learning_rate,
                                        })

            self.W_val.append(self.sess.run(self.weight))
            self.cost_val.append(self.sess.run(self.cost,
                                               feed_dict={self.learning_x_data: self.input_x_data,
                                                                     self.learning_y_data: self.input_y_data}))

            if step % 20 == 0:
                print("step:",  step,
                      " weight:", self.sess.run(self.weight),
                      " cost:", self.sess.run(self.cost,
                                               feed_dict={self.learning_x_data: self.input_x_data,
                                                          self.learning_y_data: self.input_y_data}))

    """
        입력받은 데이터를 그래프로 보여주든가 데이터를 입력받아 학습을 진행하고 나온 결과를 출력할 때 사용한다.
    """

    def show_input_data(self):
        if self.feature == "one_variable":
            plt.plot(self.input_x_data, self.input_y_data, 'ro')
            plt.plot(self.input_x_data, self.sess.run(self.weight) * self.input_x_data + self.sess.run(self.bias), label='h')
            plt.legend()
            plt.show()

        # elif self.feature == "multi_variable":
        #     pass

        else:
            text_error1 = "do not run show_input_data(self) method"
            text_error2 = "because graph is possible to show 2 dimension graph"
            print(text_error1 + "\n" + text_error2)

    def show_test_data(self):
        if self.feature == "one_variable":
            plt.plot(self.test_input_x_data, self.test_data, 'ro')
            plt.plot(self.test_input_x_data, self.sess.run(self.weight) * self.test_input_x_data + self.sess.run(self.bias), label='h')
            plt.legend()
            plt.show()

        # elif self.feature == "multi_variabl":
        #     pass

        else:
            text1 = "do not run show_input_data(self) method"
            text2 = "because graph is possible to show 2 dimension graph"
            print(text1 + "\n" + text2)

    def show_cost_data(self):
        plt.plot(self.W_val, self.cost_val, 'ro')
        plt.xlabel("weight")
        plt.ylabel("cost")
        plt.legend()
        plt.show()



if __name__ == '__main__':

    # 이 부분은 one variable linear regression을 진행했을 때 소스

    # kihoon = linearRegression("one_variable")
    # kihoon.set_data_default()
    # kihoon.learning(0.1)
    # kihoon.show_input_data()
    # kihoon.show_cost_data()
    #
    # kihoon.set_test_data_default()
    # kihoon.show_test_data()

    # one variable linear regerssion finish

    narae = linearRegression("multi_variable")
    narae.set_data_default()
    narae.learning(0.1)
    narae.show_input_data()

    # narae.set_test_data_default()

