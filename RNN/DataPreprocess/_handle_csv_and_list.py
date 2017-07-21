import csv


def _csv_to_list(_file_name):
    '''
    csv파일을 읽어서 리스트로 만든다
    :param _file_name:
    :return: csv를 리스트로 바꾼 결과물
    '''
    combine = []

    with open(_file_name) as csvfile:
        read_merged_1 = csv.reader(csvfile, delimiter=',')

        for row in read_merged_1:
            # print(row)
            combine.append(row)

    return combine

def _combine_two_list(_one_list, _two_list):
    '''
    두 개의 리스트를 하나의 리스트로 합쳐준다
    두 개의 리스트가 같은 길이를 갖고있다고 가정한다
    이 때 뒤에 리스트가 앞의 리스트의 맨 뒤에 합쳐지게 한다
    이는 1,2,3 과 111이 있다고 가정했을 때 1,2,3,111로 만들어준다
    :param _one_list:
    :param _two_list:
    :return: 합쳐진 리스트
    '''
    combine = []

    if len(_one_list) == len(_two_list):

        for i in range(len(_one_list)):
            combine.append(_one_list[i])

        for i in range(len(combine)):
            combine[i] += _two_list[i]

    else:
        print("Incorrect value length")

    return combine

def _make_list_to_csv(_wanna_object, _file_name):
    '''
    리스트를 csv파일로 바꿔준다
    :param _wanna_object: csv 파일로 바뀌길 바라는 list 오브젝트
    :param _file_name: 결과물 csv의 이름
    '''
    with open(_file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i in range(len(_wanna_object)):
            writer.writerow(_wanna_object[i])


_file_one = 'merged1.csv'
_file_two = 'merged2.csv'

_one = _csv_to_list(_file_name=_file_one)
_two = _csv_to_list(_file_name=_file_two)

_combine = _combine_two_list(_one_list=_one, _two_list=_two)

_make_list_to_csv(_wanna_object=_combine, _file_name='merged.csv')