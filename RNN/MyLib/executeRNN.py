from RNN.MyLib.lib_rnn import RNNLibrary
from RNN.MyLib.lib_dataset import Database
'''
RNN
1. setParams(input_dimension, sequence_length, output_dimension)
2. setPlaceholder(
'''

'''
Dataset


'data-02-stock_daily.csv'
'''
class DB(Database):
    def init_dataset(self):
        self.reverse()
        self.nomalization()

class RNN(RNNLibrary):
    def init_rnn_library(self):
        self.setParams(seq_length=7, data_dim=5, output_dim=1)
        self.setPlaceholder(seq_length=rnn.seq_length, data_dim=rnn.data_dim)
        self.setHypothesis(hidden_dim=10)
        self.setCostfunction()
        self.setOptimizer(0.01)


if __name__ == '__main__':
    db = DB()
    db.load('data-02-stock_daily.csv', seq_length=7)

    rnn = RNN()
    rnn.learning(db.trainX, db.trainY, loop=500)
    rnn.showErrors()
    rnn.prediction(db.testX, db.testY)

