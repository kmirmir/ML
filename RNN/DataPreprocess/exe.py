from RNN.DataPreprocess._handle_csv_and_list import _csv_to_list, _make_list_to_csv

filename = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/2014combine.csv'
result = _csv_to_list(_file_name=filename)

newly = []
for i in range(len(result)):
    print(result[i][2:])
    newly.append(result[i][2:])

print(newly)
_make_list_to_csv(_wanna_object=newly, _file_name='2014combine_to_remove_feature.csv')