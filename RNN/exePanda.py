import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# epoch, layer, learning_rate, rnn.errors[-1], rnn.rmse_val, hidden_dim

# learning rate가 0.01 일 때의 x 좌표 epoch와 y 좌표 loss 찍기
# learning rate가 0.01 와 같은 epoch 일 때의 x 좌표 layer와 y 좌표 loss 찍기

# lr = pd.DataFrame()
# 학습률이 0.01이고 epoch이 1일 때의 데이터들을 가져다 박아버림

def dataToGraphXeLayerYeLoss(epoch):
    raw_data = pd.read_csv("/Users/masinogns/PycharmProjects/rnn/totalHour.csv"
                           , names=['epoch', 'layer', 'learning_rate', 'loss', 'rmse', 'hidden'])

    save_file_name = 'sameEpoch'+str(epoch)+'XeLayerYeLoss.png'

    lr1 = raw_data[(raw_data['learning_rate'] == 0.0001) & (raw_data['epoch'] == epoch)]
    lr2 = raw_data[(raw_data['learning_rate'] == 0.001) & (raw_data['epoch'] == epoch)]
    lr3 = raw_data[(raw_data['learning_rate'] == 0.01) & (raw_data['epoch'] == epoch)]
    lr4 = raw_data[(raw_data['learning_rate'] == 0.1) & (raw_data['epoch'] == epoch)]
    # print(lr1.head(10))
    fig, ax1 = plt.subplots()
    ax1.plot(lr1.layer, lr1.loss, label='learning_rate = 0.0001')
    ax1.plot(lr2.layer, lr2.loss, label='learning_rate = 0.001')
    ax1.plot(lr3.layer, lr3.loss, label='learning_rate = 0.01')
    ax1.plot(lr4.layer, lr4.loss, label='learning_rate = 0.1')
    ax1.set_xlabel('The number of layer')
    ax1.set_ylabel('Loss')
    plt.legend(loc='upper left')
    # plt.show()

    plt.savefig(save_file_name)
    plt.close('all')



def dataToGraphXeEpochYeLoss(layer):
    raw_data = pd.read_csv("/Users/masinogns/PycharmProjects/rnn/totalHour.csv"
                           , names=['epoch', 'layer', 'learning_rate', 'loss', 'rmse', 'hidden'])

    save_file_name = 'sameLayer'+str(layer)+'XeEpochYeLoss.png'

    lr1 = raw_data[(raw_data['learning_rate'] == 0.0001) & (raw_data['layer'] == layer)]
    lr2 = raw_data[(raw_data['learning_rate'] == 0.001) & (raw_data['layer'] == layer)]
    lr3 = raw_data[(raw_data['learning_rate'] == 0.01) & (raw_data['layer'] == layer)]
    lr4 = raw_data[(raw_data['learning_rate'] == 0.1) & (raw_data['layer'] == layer)]
    print(lr1.head(10))
    fig, ax1 = plt.subplots()
    ax1.plot(lr1.epoch, lr1.loss, label='learning_rate = 0.0001')
    ax1.plot(lr2.epoch, lr2.loss, label='learning_rate = 0.001')
    ax1.plot(lr3.epoch, lr3.loss, label='learning_rate = 0.01')
    ax1.plot(lr4.epoch, lr4.loss, label='learning_rate = 0.1')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    plt.legend(loc='upper left')
    # plt.show()

    plt.savefig(save_file_name)
    plt.close('all')

for i in range(1, 6):
    dataToGraphXeLayerYeLoss(i)

for i in range(1, 8):
    dataToGraphXeEpochYeLoss(i)
