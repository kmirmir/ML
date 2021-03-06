from abc import abstractmethod
import numpy as np


class Database:
    data = []
    trainX = []
    trainY = []
    validationX = []
    validationY = []
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
        self.data = np.loadtxt(file_name, delimiter=',')

        self.init_dataset()

        x = self.data[:]
        y = self.data[:, [8]]  # Close as label
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

    def load_train_data(self, file_name = None, seq_length=None):
        self.data = np.loadtxt(file_name, delimiter=',')

        self.init_dataset()

        x = self.data[:]
        y = self.data[:, [8]]  # Close as label

        # build a dataset
        dataX = []
        dataY = []

        for i in range(0, len(y) - seq_length):
            _x = x[i:i + seq_length]
            _y = y[i + seq_length]  # Next close price
            # print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)

        self.trainX = np.array(dataX)
        self.trainY = np.array(dataY)

    def load_test_data(self, file_name = None, seq_length=None):
        self.data = np.loadtxt(file_name, delimiter=',')

        self.init_dataset()

        x = self.data[:]
        y = self.data[:, [8]]  # Close as label

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
        validation_size = int(len(dataY) * 0.5)
        test_size = len(dataY) - validation_size
        self.validationX, self.testX = np.array(dataX[0:validation_size]), np.array(
            dataX[validation_size:len(dataX)])
        self.validationY, self.testY = np.array(dataY[0:validation_size]), np.array(
            dataY[validation_size:len(dataY)])