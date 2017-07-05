from urllib.request import urlopen

import sys
from bs4 import BeautifulSoup

#         sys.stdout.write(result[i].get_text() + ", ")
#         sys.stdout.flush()

# wantYear = "2012"
url = "http://www.kma.go.kr/weather/climate/past_table.jsp?stn=184&yy=2012&obs=90&x=29&y=14"
print(url)
html = urlopen(url)
bsObject = BeautifulSoup(html.read(), "html.parser")
table = bsObject.find("table", {"class": "table_develop"})
cells = []

for row in table.findAll("tr"):
    cells.append(row.findAll("td"))

'''
 if yy == 2012, 2012년도의 날씨들을 가져오는 것
 :param a 월을 나타낸다
 :param i 일을 나타낸다
 a 월 i 일 ok?
'''
for a in range(1,13):
    print("======"+str(a)+"월======")

    for i in range(1, 32):
        # 1 월달 날씨들
        if ord(cells[1][6].get_text()) == 160:
            print("맑음")
        else:
            print(str(i)+"일 :"+cells[i][a].get_text())
