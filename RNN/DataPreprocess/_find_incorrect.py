import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from RNN.DataPreprocess._handle_csv_and_list import _csv_to_list

file_name_real = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/the_data.csv'
file_name = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/_correct_data.csv'
file_name2 = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/_exprience_data_July.csv'
the_data_part_1 = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/the_data_part_1.csv'
the_data_part_2 = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/the_data_part_2.csv'
the_data_combined = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/the_data_combined.csv'
the_data = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/2013data.csv'
a_day = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/dataToDay.csv'
four_hour = '/Users/masinogns/PycharmProjects/ML/RNN/MyLib/dataToFourHour.csv'
aaa = '/Users/masinogns/PycharmProjects/ML/RNN/finishData.csv'
_the_data = _csv_to_list(_file_name=file_name
                         )

_the_count = []


def _show_data(file_name):
    raw_data = pd.read_csv(file_name, names=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # print(raw_data.shape)
    # print(raw_data)
    # print(raw_data['1'])
    # print(raw_data['2'])
    fig, ax1 = plt.subplots()
    # fig, ax2 = plt.subplots()

    # ax1.plot(raw_data['1'], label="time")#시간
    ax1.plot(raw_data['3'], label="solar energy")#일사량
    # ax1.plot(raw_data['6'], label="temperature")#온도
    # ax1.plot(raw_data['9'], label="output")#출력량
    plt.legend(loc='upper left')
    plt.show()


# _show_data(file_name)
# _show_data(the_data_part_1)
_show_data('/Users/masinogns/PycharmProjects/ML/RNN/the_data_origin_day.csv')