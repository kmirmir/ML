from RNN.MyLib.lib_rnn import RNNLibrary
from RNN.MyLib.lib_dataset import Database

class DB(Database):
    def init_dataset(self):
        self.nomalization()

class RNN(RNNLibrary):
    def init_rnn_library(self):
        self.setParams(seq_length=24, input_dim=15, output_dim=1)
        self.setPlaceholder(seq_length=rnn.seq_length, input_dim=rnn.input_dim)
        self.setHypothesis(hidden_dim=hidden_dim, layer=layer)
        self.setCostfunction()
        self.setOptimizer(learning_rate=learning_rate)

# 일사량은 단위면적이 단위시간에 받는 일사에너지의 양으로 정의되며, 순간 복사량과 정해진 시간동안 단위면적이 받는 총열량인 복사량으로 표현됩니다.
# 날짜,시간,수평일사량,경사일사량,외기온도,모듈온도,VCB 출력,ACB 출력,인버터 츨력
# VCB = 발전기 출력 차단기 Vaccum Circuit Breacker
# ACB = 기중 차단기 Air Circuit Breacker
# 출력은 인버터 출력으로 함
# feed value of shape (655, 24, 15)
hidden_dim = 10
layer = 1
learning_rate = 0.01
epoch = 62

if __name__ == '__main__':

    path = '/Users/masinogns/PycharmProjects/ML/RNN/'
    load_file_name = '/dataToFourHour.csv'

    '''
    만약 name 을 쓴다면 weather가 들어있는 organize_plus_weather를 써야한다
    name2 를 쓴다면 weather가 들어있지 않은 organize를 써야한다
    '''

    name = 'weatherYes'
    name2 = 'weatherNo'

    save_error_file_name = 'error/' + name \
                            + 'lr' + str(learning_rate) + 'layer' + str(layer) + 'epoch' + str(epoch) + '.png'
    save_predict_file_name = 'predict/' + name \
                             + 'lr' + str(learning_rate) + 'layer' + str(layer) + 'epoch' + str(epoch) + '.png'
    save_validation_file_name = 'validation/'+ name \
                             + 'lr' + str(learning_rate) + 'layer' + str(layer) + 'epoch' + str(epoch) + '.png'
    save_csv_file_name = 'output/'+ name +'epoch' + str(epoch) + '.csv'

    load_path = '/Users/masinogns/PycharmProjects/ML/RNN/Data/'

    '''
    weather yes
    '''
    organize_plus_weather_train = 'organize_plus_weather_train.csv'
    organize_plus_weather_test = 'organize_plus_weather_test.csv'

    train_file = load_path + organize_plus_weather_train
    test_file = load_path + organize_plus_weather_test

    '''
    weather no
    '''
    organize_train = 'organize_train.csv'
    organize_test = 'organize_test.csv'

    # train_file = load_path + organize_train
    # test_file = load_path + organize_test

    db = DB()
    # db.load(correct_file, seq_length=24)
    db.load_train_data(train_file, seq_length=24)
    db.load_test_data(test_file, seq_length=24)

    rnn = RNN()
    rnn.learning(db.trainX, db.trainY, db.validationX, db.validationY, total_epoch=epoch, check_step=100)
    rnn.showErrors(error_save_filename=path + save_error_file_name)
    rnn.validation(validation_save_filename = path + save_validation_file_name)
    rnn.prediction(db.testX, db.testY, predict_save_filename=path + save_predict_file_name)

    print("train sse: {}".format(rnn.train_errors[-1]))
    print("validation sse: {}".format(rnn.validation_errors[-1]))
    print("test sse: {}".format(rnn.test_loss))
    import csv
    f = open(path+save_csv_file_name, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    # Learning rate, the number of layer, hidden_dimension, loss
    wr.writerow([epoch, layer, learning_rate, rnn.train_errors[-1], rnn.validation_errors[-1], rnn.test_loss])
    f.close()