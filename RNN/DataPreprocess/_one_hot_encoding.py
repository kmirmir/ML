from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']


def _onehot_data(data):
    values = array(data)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    #
    # print(len(onehot_encoded))
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[4, :])])
    # print(inverted)

    return onehot_encoded

# _onehot_data(data)

'''

'''


# 목록 뽑아놓은거
list = ["비", "소낙눈", "박무", "연무", "소나기", "황사", "안개", "천둥", "폭풍", "햇무리", "뇌전","번개", "달무리", "채운", "싸락눈","무지개", "소낙성진눈깨", "달코로나", "해코로나"]
# list = ["비","박무","연무","소나기"]
list_onehot_data = _onehot_data(list)
print(list_onehot_data)
element = list[3]
print(element)

if element in list:
    list_onehot_data[list.index(element)]
    # print(list.index(element))
    # print(list_onehot_data[list.index(element)])

