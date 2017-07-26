from RNN.MyLib.lib_rnn import RNNLibrary
from RNN.MyLib.lib_dataset import Database



class DB(Database):
    def init_dataset(self):
        self.nomalization()

class RNN(RNNLibrary):
    def init_rnn_library(self):
        self.setParams(seq_length=7, input_dim=7, output_dim=1)
        self.setPlaceholder(seq_length=rnn.seq_length, input_dim=rnn.input_dim)
        self.setHypothesis(hidden_dim=hidden_dim, layer=layer)
        self.setCostfunction()
        self.setOptimizer(learning_rate=learning_rate)

# 일사량은 단위면적이 단위시간에 받는 일사에너지의 양으로 정의되며, 순간 복사량과 정해진 시간동안 단위면적이 받는 총열량인 복사량으로 표현됩니다.
# 날짜,시간,수평일사량,경사일사량,외기온도,모듈온도,VCB 출력,ACB 출력,인버터 츨력
# VCB = 발전기 출력 차단기 Vaccum Circuit Breacker
# ACB = 기중 차단기 Air Circuit Breacker
# 출력은 인버터 출력으로 함
import sys
var1 = sys.argv[1]
var2 = sys.argv[2]


epoch = 5
hidden_dim = 10


layer = int(var1)
learning_rate = 0.0001 * pow(10, int(var2))

if __name__ == '__main__':
    path = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib'
    load_file_name = '/dataToFourHour.csv'
    # save_error_file_name = '/error/' + 'error' + 'lr' + 'epoch' + str(epoch) +'layer' + str(
    #     layer) +  str(learning_rate) + 'hidden' + str(hidden_dim) +  '.png'
    # save_predict_file_name = '/predict/' + '/predict' + 'epoch' + str(epoch) + 'layer' + str(
    #     layer) + 'lr' + str(learning_rate) + 'hidden' + str(hidden_dim) + '.png'
    #
    save_error_file_name = '/error/' + 'epoch' + str(epoch) + '/layer' + str(layer) + '/error' + 'lr' + str(
        learning_rate) + 'layer' + str(layer) + 'hidden' + str(hidden_dim) + 'epoch' + str(epoch) + '.png'
    save_predict_file_name = '/predict/' + 'epoch' + str(epoch) + '/layer' + str(layer) + '/predict' + 'lr' + str(
        learning_rate) + 'layer' + str(layer) + 'hidden' + str(hidden_dim) + 'epoch' + str(epoch) + '.png'

    save_csv_file_name = '/output/' + 'epoch' + str(epoch) + '.csv'

    db = DB()
    db.load(path+load_file_name, seq_length=7)

    rnn = RNN()
    rnn.learning(db.trainX, db.trainY, loop=1000, total_epoch=epoch, check_step=100)
    rnn.showErrors(error_save_filename=path+save_error_file_name)
    rnn.prediction(db.testX, db.testY, predict_save_filename=path+save_predict_file_name)


    import csv
    f = open(path+save_csv_file_name, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    # Learning rate, the number of layer, hidden_dimension, loss
    wr.writerow([epoch, layer, learning_rate, rnn.errors[-1], rnn.test_loss, hidden_dim])
    f.close()

    print("Epoch : {}".format(epoch))
    print("Layer : {}".format(layer))
    print("Learning rate : {}".format(learning_rate))
    print("Loss : {}".format(rnn.errors[-1]))
    print("It is done")