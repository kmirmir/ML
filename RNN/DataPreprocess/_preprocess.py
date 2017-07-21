from RNN.DataPreprocess._crawling_weather import printResultOfWeather, urlMaker, crawlingWeather
from RNN.DataPreprocess._handle_csv_and_list import _make_list_to_csv, _csv_to_list
from RNN.DataPreprocess._one_hot_encoding import _onehot_data


def _check_value(list):
    result = []
    weather = ["맑음", "비", "박무", "연무", "소나기", ""]
    list_onehot_data = _onehot_data(weather)

    print("=============")
    print("Preprocessing")
    print(weather)
    print(list_onehot_data)
    print("=============")

    for a in range(len(list)):
        element = list[a]
        # print(a)
        # print(element)
        if element in weather:
            # print(weather.index(element))
            print(list_onehot_data[weather.index(element)])
            result.append(list_onehot_data[weather.index(element)])
        else:
            # print("기타")
            print(list_onehot_data[5])
            result.append(list_onehot_data[5])

        # print(list.index(element))
        # print(list_onehot_data[list.index(element)])

    return result


'''
1. 리스트 안에 있는지 체크한다
2. 리스트 안에 밸류가 있다면 해당 리스트 밸류의 인덱스를 받아놓는다
3. 원 핫 인코딩된 리스트에 인덱스를 넣어서 원 핫 인코딩으로 바꾼다
4. 끝
'''


def _exe_():
    url = urlMaker(year=2013)
    cells = crawlingWeather(url)
    result = printResultOfWeather(cells)
    print(result)
    test1 = ['황사', '황사', '맑음', '연무', '맑음', '맑음', '비', '비박무']
    test = ["맑음", "연무", "비"]
    rest = _check_value(result)
    _make_list_to_csv(_wanna_object=rest, _file_name='2013weathet.csv')


# exe()

_file_2013 = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/2013weathet.csv'
_file_2014 = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/2014weathet.csv'

weather2013 = _csv_to_list(_file_2013)
_convert_weather_2013 = []

for i in range(len(weather2013)):
    for _a_day in range(24):
        _convert_weather_2013.append(weather2013[i])

print(_convert_weather_2013)
_make_list_to_csv(_wanna_object=_convert_weather_2013, _file_name='2013convertweather.csv')

print(len(weather2013))
print(len(weather2013*24))

weather2014 = _csv_to_list(_file_2014)
_convert_weather_2014 = []

for i in range(len(weather2014)):
    for _a_day in range(24):
        _convert_weather_2014.append(weather2014[i])

_make_list_to_csv(_wanna_object=_convert_weather_2014, _file_name='2014convertweather.csv')