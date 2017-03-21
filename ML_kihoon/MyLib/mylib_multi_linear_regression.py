#-*- coding: utf-8 -*-
# tensor flow
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os

class multi_linear():
    train_x = []
    train_y = []

    # 플레이스홀더 처리
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    w = None
    sess = None
    init = None

    W_val = []
    cost_val = []

    def __init__(self, x_data, y_data):
        self.train_x = x_data
        self.train_y = y_data

        # 변수 랜덤값 설정(1*n 행렬로 설정, b를 없애기 위해 [1, 1, 1, 1, 1]추가함)
        self.w = tf.Variable(tf.random_uniform([1, len(self.train_x)], -1.0, 1.0))

        self.sess = tf.Session()
        self.init = tf.initialize_all_variables()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def model_running(self, show_or_hidden, loop_size, print_point):
        # 가설 공식 적용(특징이 여러개이기 때문에 w * X의 행렬의 곱셈을 해줘야함)
        self.hypothesis = tf.matmul(self.w, self.X)


        # 비용 계산, gradientdescent algorithm
        cost = tf.reduce_mean(tf.square(self.hypothesis-self.train_y))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train = optimizer.minimize(cost)



        self.sess.run(self.init)

        if show_or_hidden == 1:
            for step in range(loop_size):
                self.sess.run(train, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                if step % print_point == 0:
                    print(step, "단계 : ",
                          "에러 : ", self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}),
                          "기울기(Weight) : ", self.sess.run(self.w))

                    # for graphs
                    self.W_val.append(self.W_val.append(self.sess.run(self.w)))
                    self.cost_val.append(self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}))


        elif show_or_hidden == 2:
            for step in range(loop_size):
                self.sess.run(train, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                if step % print_point == 0:
                    self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                    # for graphs
                    self.W_val.append(self.W_val.append(self.sess.run(self.w)))
                    self.cost_val.append(self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}))

        else:
            pass



    def test_running(self, test_x):

        # 결과예측(5개의 트레이닝 셋을 연습시켜서 w,b 값을 얻어냈고, 이제 이것을 바탕으로 예측하기 위해 가설값으로 예측해본다.(1, 특징1, 특징2)로
        print (self.sess.run(self.hypothesis, feed_dict={self.X: test_x}))
        print (self.sess.run(self.hypothesis, feed_dict={self.X: [[1], [2], [2]]}))

    def show_cost(self):
        plt.plot(self.cost_val, 'o-', label='cost')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # 데이터 특징이 2개인 것(n * 1 행렬로 표현)
    x_data = [[1, 1, 1, 1, 1],
         [1, 0, 3, 0, 5],
         [0, 2, 0, 4, 0]]

    y_data = [1, 2, 3, 4, 5]


    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = np.loadtxt(script_dir + '/train.txt', unpack=True, dtype='float32')
    # train data set ver 2
    x_data2 = train_data[0:-1]
    y_data2 = train_data[-1]

    # example data
    test_x = [[1, 1, 1, 1, 1],
          [5, 4, 3, 2, 1],
          [5, 4, 3, 2, 1]]




    kihoon = multi_linear(x_data, y_data)
    kihoon.set_learning_rate(0.1)
    kihoon.model_running(1, 2001, 20)
    kihoon.test_running(test_x)
    kihoon.show_cost()