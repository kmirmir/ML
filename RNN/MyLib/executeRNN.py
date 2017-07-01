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
        # self.reverse()
        self.nomalization()

class RNN(RNNLibrary):
    def init_rnn_library(self):
        self.setParams(seq_length=24, input_dim=9, output_dim=1)
        self.setPlaceholder(seq_length=rnn.seq_length, input_dim=rnn.input_dim)
        self.setHypothesis(hidden_dim=10)
        self.setCostfunction()
        self.setOptimizer(0.01)

# 날짜,시간,수평일사량,경사일사량,외기온도,모듈온도,VCB출력,ACB출력,인버터츨력
# 출력은 인버터 출력으로 함

if __name__ == '__main__':
    db = DB()
    db.load('/Users/masinogns/PycharmProjects/ML/RNN/MyLib/finishData.csv', seq_length=24, month=2)

    rnn = RNN()
    rnn.learning(db.trainX, db.trainY, loop=500, check_step=100)
    rnn.showErrors()
    rnn.prediction(db.testX, db.testY)

