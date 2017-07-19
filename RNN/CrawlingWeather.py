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
    # print("url making : " + url)

    return url

def crawlingWeather(url):
    # print("result of url maker : " + url)
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
    '''
    for a in range(1, 13):  # 1월부터 12월까지 반복문을 실행한다
        print("======" + str(a) + "월======")

        for i in range(1, 32):  # 1일부터 31일까지 반복문을 실행한다
            # 1 월달 날씨들
            # print(cells[1][6].get_text())
            compareString = cells[i][a].get_text()
            if ord(compareString[0]) != 160:
                print(str(i) + "일 :" + cells[i][a].get_text())
            else:
                print(str(i) + "일 :맑음")

def findFeatureOfWeather(cells):
    feature = []

    for a in range(1, 13):  # 1월부터 12월까지 반복문을 실행한다
        # print("======" + str(a) + "월======")
        for i in range(1, 32):  # 1일부터 31일까지 반복문을 실행한다
            get_text = cells[i][a].get_text()
            # print(str(i) + "일 :" + get_text)

            for i in range(len(feature)):
                if feature[i] != get_text:
                    feature.append(get_text)

    print("finish find feature weather")
    for i in range(len(feature)):
        print(feature[i])

    return feature

def printResultOfFind(feature):
    for i in range(len(feature)):
        print(feature[i])

url = urlMaker(year=2012)
cells = crawlingWeather(url)
printResultOfWeather(cells)

# feature = findFeatureOfWeather(cells)
# printResultOfFind(feature)


