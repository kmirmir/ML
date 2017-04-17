from MyLib.mylib_softmax_classification import softmax

'''
x data = 매움의 정도 따듯함, 밥 유무 가격의 정도 고기의 유무로 소프트맥스를 실행함
y data = 공책에 국박 해장국 갈비찜 짜장 고기 짬뽕, 치킨을 분류한 것
'''
x_data = [
    [0,0,0,1,0],
    [1,0,0,1,0],
    [0,0,0,1,1],
    [0,1,1,0,0],
    [2,0,1,0,1],
    [0,0,1,2,1]
]

y_data = [
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1]
]

feed_dict = [
    [0,0,0,0,0],
    [0,0,0,0,1],
    [0,1,1,0,0]
]
so = softmax(x_data, y_data)
so.run()
so.predict(feed_dict)
