import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup


def urlMaker(year):
    '''
    url에는 요소가 3개가 있다
    지점 연도 요소
    지점은 지역을 정할 수 있다
    연도는 연도별 자료
    요소는 날씨 강수량 평균풍속 ... 이 있다

    stn 184는 제주 지역을 의미한다
    yy 2012는 2012년도를 의미한다
    obs=90&x=28&y=10은 날씨 요소를 obs=07&x=21&y=8은 평균기온을 의미한다

    :return: url
    '''
    domainName = "http://www.kma.go.kr/weather/climate/past_table.jsp?"
    year = "yy="+str(year)
    identify = "stn=184" \
               "&" \
               + year +\
               "&" \
               "obs=90&x=29&y=14"

    url = domainName + identify

    return url

def crawlingWeather(url):
    html = urlopen(url)
    bsObject = BeautifulSoup(html.read(), "html.parser")
    table = bsObject.find("table", {"class": "table_develop"})
    cells = []

    for row in table.findAll("tr"):
        cells.append(row.findAll("td"))

    return cells


def printResultOfWeather(cells):
    '''
     if yy == 2012, 2012년도의 날씨들을 가져오는 것
     :param a 월을 나타낸다
     :param i 일을 나타낸다
     a 월 i 일 ok?

     2017/7/20 : 이제 원핫인코딩으로 바꿔
    '''

    _month_2013 = ['31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']

    result = []
    for a in range(1, 13):  # 1월부터 12월까지 반복문을 실행한다
        # print("======" + str(a) + "월======")
        for i in range(1, int(_month_2013[a-1])+1):  # 1일부터 31일까지 반복문을 실행한다
            compareString = cells[i][a].get_text()
            if ord(compareString[0]) != 160:
                print(str(a) +"월 "+ str(i) + "일 :" + cells[i][a].get_text())
                result.append(cells[i][a].get_text())
            else:
                print(str(a) +"월 "+ str(i) + "일 :맑음")
                result.append("맑음")
    return result

def findFeatureOfWeather(cells):
    '''
    날씨 정보에서 feature들을 빼 내려고 만든 함수
    feature 리스트에 없는 값은 더하고 리스트에 값이 있다면 계속 진행한다

    :param cells:
    :return:
    '''
    feature = []
    month = ['31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']

    for a in range(0, 12):  # 1월부터 12월까지 반복문을 실행한다
        # print("======" + str(a) + "월======")
        for i in range(1, month[a]):  # 1일부터 31일까지 반복문을 실행한다
            get_text = cells[i][a].get_text()
            # print(str(i) + "일 :" + get_text)

            if get_text in feature:
                pass
            else:
                feature.append(get_text)

    print("finish find feature weather")

    return feature

def organizeFeatureOfWeather(feature):
    # 목록 뽑아놓은 거
    '''
    뽑아놓은 list 목록에 있는 값이 있다면 공뱅으로 replace시킨다
    만약 출력에 아무것도 안나온다면 feature 뽑기가 제대로 된 것이다
    :param feature:
    :return:
    '''
    list = ["비", "소낙눈", "박무", "연무", "소나기",
            "황사", "안개", "천둥", "폭풍", "햇무리", "뇌전",
            "번개", "달무리", "채운", "싸락눈",
            "무지개", "소낙성진눈깨", "달코로나", "해코로나"]

    for i in range(len(feature)):
        for a in range(len(list)):
            if list[a] in feature[i]:
                feature[i] = feature[i].replace(list[a], "")

    return feature

def printResultOfFindFeature(feature):
    for i in range(len(feature)):
        print(feature[i])

def writeCsv(filename, list):
    f = open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range(len(list)):
        wr.writerow(list[i])
    f.close()




# url = urlMaker(year=2014)
# cells = crawlingWeather(url)
# printResultOfWeather(cells)

# feature = findFeatureOfWeather(cells)
# printResultOfFindFeature(feature)
# feature = organizeFeatureOfWeather(feature)
# printResultOfFindFeature(feature)

