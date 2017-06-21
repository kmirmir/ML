# 신경망 클래스 정의
class neuralNetwork:

    # 신경망 초기화하기
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 입력 은닉 출력 계층 노드 개수 설정
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 학습률
        self.learningrate = learningrate

    # 신경망 학습시키기
    def train(self):
        pass

    # 신경망에 질의하기
    def query(self):
        pass

    # 신경망 코스트 그림으로 보기
    def show_cost_graph(self):
        pass

    # 신경망 코스트 콘솔에 출력하기
    def show_cost_console(self):
        pass
