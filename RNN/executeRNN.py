
'''
RNN
1. setParams(input_dimension, sequence_length, output_dimension)
2. setPlaceholder(
'''
from RNN.MyRNN2 import Database, RNNLibrary

'''
Dataset

1. loadDataset() 꼭 실행되어야 하는 것
2. reverse() 옵션
3. nomalization() 옵션
4. rateOfTrainDataset() 옵션 기본 70

'data-02-stock_daily.csv'
'''
class DB(Database):
    def init_dataset(self):
        self.xy = self.nomalization(self.xy)

if __name__ == '__main__':
    db = DB()
    db.load(7)

    rnn = RNNLibrary()
    rnn.setPlaceholder(seq_length=rnn.seq_length, data_dim=rnn.data_dim)
    rnn.run(db.trainX, db.trainY, db.testX, db.testY)

