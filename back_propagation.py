import numpy as np
import matplotlib.pyplot as plt

class MLP(object):
    
    @staticmethod # 정적 메소드
    def sigmoid(x): # 활성함수 sigmoid
        return 1/(1+np.power(np.e, -x)) # 1/1+e^-x

    @staticmethod
    def identity(x): # static 변수로 변경
        return x

    @staticmethod
    def inv_sig(x):
        tmp = MLP.sigmoid(x)
        return tmp*(1-tmp)

    def __init__(self, MODEL_LAYER): ## **연산자 사용
        self.layers = MODEL_LAYER # 모델의 LAYER
        self.activate = [MLP.identity] # 각 층의 적용되는 활성함수가 들어가는 리스트
        self.weights = [1] # 웨이트 값
        self.bias = [0] # 바이어스 값
        self.lr = 0.002 # 학습률

        for i in range(1, len(self.layers)): # 각 층에 웨이트 값과 바이어스 값을 랜덤으로 지정)
            self.weights.append(np.random.normal(0, 0.5, (self.layers[i-1],self.layers[i]))) # 평균 0, 표준편차 0.5, layer의 i-1번째 노드수 x i번째 노드수 형태의 행렬로 랜덤 웨이트 지정
            self.bias.append(np.random.normal(0, 0.5, self.layers[i])) # 평균 0, 표준편차 0.5, layer의 i번째 노드 수만큼 bias 랜덤 지정
            # weights 개수 형태 = [[1], [5], [5x5], [5x5], [5]]
            # bias 개수 형태 = [[1], [5], [5], [5], [1]]

            if i != len(self.layers) -1: # 활성함수 형태 = [identity, sigmoid, sigmoid, sigmoid, identity] identity = 입력층, 출력층은 활성함수가 없음
                self.activate.append(MLP.sigmoid)
            else:
                self.activate.append(MLP.identity)

    def feed_forward(self, xs): # 정현파, xs값 입력

        self.U = [xs]
        self.Z = [xs]
        for i in range(1, len(self.layers)):
            u = self.Z[i-1].dot(self.weights[i]) + self.bias[i] # 행렬 곱 dot함수 사용 (xs) X (웨이트값) + (바이어스)
            z = self.activate[i](u) # u 값에 활성함수 sigmoid 적용
            self.U.append(u)
            self.Z.append(z)

        return self.Z[-1] # Z의 가장 마지막 값 = feed_forward를 거친 마지막 출력값


    def back_propagate(self, xs, ys): # 학습 데이터 xs값과 라벨 ys값 입력

        pred_ys = self.feed_forward(xs) # feed_forward의 출력값 (예측한 결과)
        self.D = []

        for i in reversed(range(1,len(self.layers))): # i = 4 3 2 1
            if i == len(self.layers) - 1:
                d = pred_ys - ys # 예측한 값 - 정답 값
            else:
# ---------------------------------------------------------------------------------------------------------잘 모르겠음
                d = self.inv_sig(self.U[i])*(self.D[-1].dot(self.weights[i+1].T))

            dW = self.Z[i-1].T.dot(d)
            db = np.sum(d, axis=0) # np.sum(axis = 0) 가장 큰 범위의 list를 제거한 후 각 리스트에 해당하는 값끼리 더함 (차원이 1개 줄어듬)
# ---------------------------------------------------------------------------------------------------------잘 모르겠음

            self.weights[i] -= self.lr*dW # weight = weight - (학습률 X weight)
            self.bias[i] -= self.lr*db # bias = bias - (학습률 X bias)
            self.D.append(d)
        return

    def evaluate(self, xs, ys):
        pred_ys = self.feed_forward(xs)
        d = pred_ys - ys
        return np.mean(np.sqrt(d**2)) # d의 제곱의 평균
    

if __name__ == '__main__':
    layers=[1,5,5,5,1]
    model = MLP(layers)
    
    train_x = np.linspace(-5,5,100) # -5부터 5까지의 수를 100개로 나눔
    train_y = np.sin(train_x) #  train_x값을 sin함수를 취함

    xs = train_x.reshape(-1,1) # 1차원으로 바꿔줌
    ys = train_y.reshape(-1,1)

    pred_ys = model.feed_forward(xs) # 순전파 함수에 xs값 입력

    for i in range(1):
        model.back_propagate(xs, ys)

        if (i+1) % 5000 == 0:
            error = model.evaluate(xs, ys)
            print('ITER={:05d}회, accuracy={:.2f}%'.format(i+1, (1-error)*100))
            
    pred_ys = model.feed_forward(xs)
    
    plt.plot(xs.ravel(), pred_ys.ravel(), label='prediction')
    plt.plot(xs.ravel(), train_y.ravel(), label='original')
    plt.legend()
    plt.show() 