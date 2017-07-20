import numpy as np


def _csv_to_list(_file_name):

    list = np.loadtxt(_file_name, dtype=str)
    result = []
    for i in range(len(result)):
        print(result[i])

    return list

_file_name = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/month.csv'
list = _csv_to_list(_file_name=_file_name)

print(list)

qqq = ['31' ,'28' ,'31' ,'30' ,'31', '30', '31', '31' ,'30' ,'31' ,'30', '31']
print(qqq)
for i in range(len(qqq)):
    print(qqq[i])