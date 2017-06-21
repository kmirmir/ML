from CNN.mylib_cnn import neuralNetwork

# 입력 은닉 출력 노드의 수
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# 학습률 정의
learning_rate = 0.3

# 신경망 객체 생성
neural = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)