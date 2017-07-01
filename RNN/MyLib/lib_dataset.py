from abc import abstractmethod
import numpy as np


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

    def load(self, file_name = None, seq_length=None, month=None):
        # Open, High, Low, Volume, Close
        self.data = np.loadtxt(file_name, delimiter=',')

        self.init_dataset()

        # 24 시간 = 하루  * 7일 = > 1 주일 * x => x 주일
        data_length = 24 * 7 * 4 * month
        x = self.data[:data_length]
        y = self.data[:data_length, [-1]]  # Close as label
        # print(x)
        # print(y)
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

        print("month : ", month)
        print("total data size : ", data_length)
        print("train data size : ", data_length * 0.7)
        print("test data size : ", data_length * 0.3)