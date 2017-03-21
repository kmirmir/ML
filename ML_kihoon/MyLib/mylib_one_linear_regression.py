# TensorFolw=Tutorials

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class linear_regression:

    train_x = []
    train_y = []

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W_val = []
    cost_val = []

    test_data = []

    init = tf.initialize_all_variables()

    sess = tf.Session()

    hypothesis = W * X + b


    def set_random_data(self, abstract_weight):
        # 데이터 랜덤하게 생성하기.
        num_points = 100
        vectors_set = []

        for i in range(num_points):
            x1 = np.random.normal(0.0, 0.55)
            y1 = x1 * abstract_weight + 0.3 + np.random.normal(0.0, 0.03)
            vectors_set.append([x1, y1])

        x_data = [v[0] for v in vectors_set]
        y_data = [v[1] for v in vectors_set]

        plt.plot(x_data, y_data, 'ro')
        plt.legend()
        plt.show()

    def set_train_data(self, x_data, y_data):
        self.train_x = x_data
        self.train_y = y_data

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # default loop size is 2001, print_point is 20, show or hidden 0 or 1;
    def model_running(self, show_or_hidden, loop_size, print_point):

        cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(cost)


        self.sess.run(self.init)


        if show_or_hidden == 1:
            for step in range(loop_size):
                self.sess.run(train, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                if step % print_point == 0:
                    print ( step, "단계 : ",
                           "에러 : ", self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}),
                           "기울기(Weight) : ", self.sess.run(self.W),
                           "bias : ",self.sess.run(self.b))

                    # for graphs
                    self.W_val.append(self.W_val.append(self.sess.run(self.W)))
                    self.cost_val.append(self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}))


        elif show_or_hidden == 2:
            for step in range(loop_size):
                self.sess.run(train, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                if step % print_point == 0:
                    self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y})

                    # for graphs
                    self.W_val.append(self.W_val.append(self.sess.run(self.W)))
                    self.cost_val.append(self.sess.run(cost, feed_dict={self.X: self.train_x, self.Y: self.train_y}))

        else:
            pass


        print("최종 학습 weight 값 : ", self.sess.run(self.hypothesis, feed_dict={self.X: 1}))
        print("최종 학습 bias 값 : ", self.sess.run(self.b))

    def test_running(self, test_x):
        for step in range(len(test_x)):
            # print(test_x[step])
            self.test_data.append(self.sess.run(self.hypothesis,
                                           feed_dict={self.X: test_x[step]}))

            print("모델의 예측 값 : ", self.test_data[step])

    def show_weight(self):
        plt.plot(self.W_val, 'o-', label='weight')
        plt.legend()
        plt.show()

    def show_cost(self):
        plt.plot(self.cost_val, 'o-', label='cost')
        plt.legend()
        plt.show()

    def show_train(self):
        plt.plot(self.train_x, self.train_y, 'ro', label='train data')
        plt.plot(self.train_x, self.sess.run(self.W)* self.train_x + self.sess.run(self.b), label='hypothesis')
        plt.legend()
        plt.show()

"""
사용방법

set_train_data(x_data, y_data)
리스트 형태의 x, y 데이터를 각 각의 파라미터로 넣어준다
]

set_learning_rate(learning_rate)
모델 학습 시의 learning rate를 조절해 줄 수 있다
]

model_running(show_or_hidden, loop_size, print_point)
show_or_hidden은 학습을 진행할 때, 학습 진행 정보를 출력할지 말지를 결정하는 파라미터다.
1일 때는 학습 정보를 출력한다. 그리고 최종 학습에서 나온 기울기와 bias 값 또한 출력한다.
2일 때는 학습 정보를 출력하지 않고 최종 학습에서 나온 기울기(weight)값과 bias값만 출력한다.

loop_size는 학습을 몇번 진행할지 결정하는 변수이다.

pirnt_point는 학습이 진행됨에 따라서 출력 구간 포인트를 정해서 loop_size만큼 반복되는 것을 방지한다.
]

test_running(test_x)
학습된 모델을 이용해서 예측 값을 출력해보는 함수이다
test_x에 list 변수를 파라미터로 넣어 각 각의 값에 해당하는 예측 값들을 출력시킨다.
]

show_train
학습에 사용된 데이터와 학습해서 나온 가설을 출력한다
]

show_weight
학습할 때 기울기의 변화를 출력한다
학습 결과로 나온 기울기를 미분했을 때 에러가 가장 낮아야 한다
만약 그렇지 못하다면 해당 모델은 학습이 제대로 이루어지지 않은 모델이다
]

show_cost
학습할 때 에러의 변화를 출력한다
에러가 0일 때가 가장 잘 학습된 모델이 된다
]
"""

'''
x = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
gildong.set_train_data(x, y)

test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

gildong.set_test_data(test_X, test_Y)
gildong.run_full()

'''

class kihoon(linear_regression):
    def __init__(self):
        x_data = [1, 2, 3]
        y_data = [1, 2, 3]
        test_x = [7, 9, 11]

        kihoon = linear_regression()
        kihoon.set_train_data(x_data, y_data)
        kihoon.set_learning_rate(0.1)
        kihoon.model_running(2, 2001, 20)

        kihoon.show_train()
        kihoon.show_weight()
        kihoon.show_cost()

        kihoon.test_running(test_x)

class narae(linear_regression):
    def __init__(self):
        self.set_random_data(0.1)
        self.set_learning_rate(0.1)
        self.model_running(1, 2001, 20)



if __name__ == '__main__':
    linear = kihoon()
    # linear2 = narae()