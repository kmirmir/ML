from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

'''
http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
I find source code How to One Hot Encode Sequence Data in Python above URI
'''
def _onehot_data(data):
    values = array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded



'''
follow code just exaple for me to understand easy
'''

def example():
    # 목록 뽑아놓은거
    # list = ["비", "소낙눈", "박무", "연무", "소나기", "황사", "안개", "천둥", "폭풍", "햇무리", "뇌전","번개", "달무리", "채운", "싸락눈","무지개", "소낙성진눈깨", "달코로나", "해코로나"]
    list = ["비", "박무", "연무", "소나기"]
    list_onehot_data = _onehot_data(list)
    print(list_onehot_data)
    element = list[3]
    print(element)
    if element in list:
        list_onehot_data[list.index(element)]
        print(list.index(element))
        print(list_onehot_data[list.index(element)])


# example()

