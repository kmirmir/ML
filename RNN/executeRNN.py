from RNN.MyRNN2 import Database, RNNLibrary
'''
RNN
1. setParams(input_dimension, sequence_length, output_dimension)
2. setPlaceholder(
'''

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
        self.data = self.nomalization(self.data)

if __name__ == '__main__':
    db = DB()
    db.load('data-02-stock_daily.csv', seq_length=7)

    rnn = RNNLibrary()
    rnn.setParams(seq_length=7, data_dim=5, output_dim=1)
    rnn.setPlaceholder(seq_length=rnn.seq_length, data_dim=rnn.data_dim)
    rnn.run(db.trainX, db.trainY, db.testX, db.testY)

