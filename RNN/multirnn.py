import numpy as np
import csv


'''
하루 24시간을 시퀀스로 넣는다. 아웃풋은 한 시간의 생산량을 예측한다.
하루 24시간의 데이터들을 평균을 낸 후에 시퀀스로 한 주(seq_lenght=7)를 넣는다. 아웃풋은 하루의 생산량을 예측한다.
한 주의 데이터들을 평균을 낸 후에 시퀀스로 한 달(seq_length=4)을 넣는다. 아웃풋은 한 주의 생산량을 예측한다.
한 달의 데이터들을 평균을 낸 후에 시퀀스로 6개월(seq_length=6)을 넣는다. 아웃풋은 한 달의 생산량을 예측한다.
'''

def getResultOfDataToSometing(something):

    data = np.loadtxt('/Users/masinogns/PycharmProjects/ML/RNN/_correct_data.csv', delimiter=',')
    x = data[:, 1:]
    '''
    something = 24 --> 하루로 묶은 것
    '''
    # something = 24
    print(len(x) / something)

    result = []
    for i in range(int(len(x) / something)):
        day = x[0 + (something * i):something + (something * i), :]
        listOfResult = []

        # 하루의 속성 한 줄씩 가져오는 것
        # 하루만 된다
        for a in range(1, 8):
            temp = 0
            # print("===start===")
            for b in range(something):
                temp += day[:, a:a + 1][b]

            # # print(temp)
            temp = float(temp)
            listOfResult.append(temp)

        result.append(listOfResult)

    return  result



def writeCsv(filename):
    global i
    f = open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range(len(result)):
        wr.writerow(result[i])
    f.close()


# 앞에는 x축(가로)을 다루고 뒤에는 y축(세로)을 다룬다
# 날짜,시간,수평일사량,경사일사량,외기온도,모듈온도,VCB 출력,ACB 출력,인버터 츨력
# 여기서 날짜를 뺀 모든 것을 가져오라고 함
'''
    something = 24 --> 하루로 묶은 것
'''
result = getResultOfDataToSometing(12)


for i in range(len(result)):
    print(result[i])
    # print("result: {}".format(result[i]))

print(len(result))


filenmae = "dataToDay.csv"


writeCsv(filename=filenmae)