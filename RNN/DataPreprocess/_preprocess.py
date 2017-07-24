from RNN.DataPreprocess._crawling_weather import printResultOfWeather, urlMaker, crawlingWeather
from RNN.DataPreprocess._handle_csv_and_list import _make_list_to_csv, _csv_to_list, _read_file_combine_one_file
from RNN.DataPreprocess._one_hot_encoding import _onehot_data

def _check_value(list):
    '''
    1. 리스트 안에 있는지 체크한다
    2. 리스트 안에 밸류가 있다면 해당 리스트 밸류의 인덱스를 받아놓는다
    3. 원 핫 인코딩된 리스트에 인덱스를 넣어서 원 핫 인코딩으로 바꾼다
    4. 끝
    '''
    result = []
    weather = ["맑음", "비", "박무", "연무", "소나기", ""]
    list_onehot_data = _onehot_data(weather)

    print("=============")
    print("Preprocessing")
    print(weather)
    print(list_onehot_data)
    print("=============")

    for i in list:
        for element in weather:
            if element in i:
                print(element)
                print(list_onehot_data[weather.index(element)])
                result.append(list_onehot_data[weather.index(element)])
                break

    return result



# 31 30 31 30 31 31 30 31 30

def _exe():
    url = urlMaker(year=2014)
    cells = crawlingWeather(url)
    result = printResultOfWeather(cells)
    print(result)
    test1 = ['황사', '황사', '맑음', '연무', '맑음', '맑음', '비', '비박무']
    test = ["맑음", "연무", "비"]
    rest = _check_value(result)

    # print("===")
    # for i in rest:
    #     print(i)
    _make_list_to_csv(_wanna_object=rest, _file_name='2014weather.csv')




def _convert_weather_multiple_24():
    _file_2013 = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/데이터/2014.csv'
    weather2013 = _csv_to_list(_file_2013)
    _convert_weather_2013 = []

    for i in range(len(weather2013)):
        for _a_day in range(24):
            _convert_weather_2013.append(weather2013[i])

    _make_list_to_csv(_wanna_object=_convert_weather_2013, _file_name='2014convertweather.csv')



def _eee():
    _file_the_data = '/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/_correct_data.csv'

    _the_data = _csv_to_list(_file_name=_file_the_data)

    for i in range(len(_the_data)):
        # print(_the_data[i])
        # print(_the_data[i][0])
        # print(_the_data[i][1])
        print(_the_data[i])



    remainder = 39 * 24
    # print(remainder)
    # _weather2013_e = _weather2013[0:len(_weather2013)-remainder]
    # print(len(_weather2013))
    # print(len(_weather2013_e))

    # for i in range(len(_weather2013_e)):
    #     print(_weather2013_e[i])

    print(len(_the_data))

# 2013년도 날씨 데이터 변환 시 나와야 하는 밸류 갯수 값
# 8136/24 = 339 , 24하면 363이 나오긴 하네
#  31+28+30+30+31+30+31+31+30+31+29+31 = 363이 나와야데는데
# 2013, 14 --> 31 28 31 30 31 30 31 31 30 31 30 31 = 365

# _exe()
# 145는 어떻게 나온거 7월 1일부터 11월 22일까지 x + 145 + 39, 181
# 31+28+31+30+31+30+31+31+30+31+30+31 = 365
# _convert_weather_multiple_24()

# _eee()


_read_file_combine_one_file(_file_one='/Users/masinogns/PycharmProjects/ML/RNN/MyLib/the_data_part_2.csv',
                            _file_two='/Users/masinogns/PycharmProjects/ML/RNN/DataPreprocess/2014convertweather.csv',
                            _file_name='2014combine.csv')


