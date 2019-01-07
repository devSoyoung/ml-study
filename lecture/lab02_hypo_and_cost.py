import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# -1과 1 사이에서 랜덤한 수 생성 - 초기값 설정
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))      # random_uniform : 정규분포 난수 생성 함수(params: 배열 모양, 최소, 최대값)
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))      # -1에서 1 사이의 난수를 1개 생성
# 난수를 사용한 이유: 최저 비용을 스스로 찾아가야 하는데 시작 위치가 매번 달라지더라도
# 항상 최저 비용을 찾는다는 것을 보여주기 위해서

hypothesis = W * x_data + b     # H(x) = Wx + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))       # 제곱 평균 오차 (RSM), 결과값은 실수 1개

rate = tf.Variable(0.1)     # 학습률, 여러 번 테스트하면서 적절한 값을 찾아야함
optimizer = tf.train.GradientDescentOptimizer(rate)     # GDO: 미분을 통해 최저비용을 향해 진행하도록 만드는 핵심 함수
# rate로 0.1을 전달했기 때문에, 매번 0.1만큼씩 내려간다(W축에 대해)
# gradient: 기울기
train = optimizer.minimize(cost)
# minimize: 기울기를 계산해서 변수에 적용하는 일을 수행
# W와 b를 적절하게 계산해서 변경하는 역할(cost가 작아지는 방향으로)

## train에 연결되는 것(학습에 필요한 것)
# optimizer, dost, rate, hypothesis, y_data, x_data, W, b

init = tf.initialize_all_variables()    # 모든 Variable 초기화 (tf.run(variables))

sess = tf.Session()
sess.run(init)      # init에 포함된 모든 텐서(=다차원 배열)를 평가

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:      # 20단계마다 출력
        print('{:4} {} {} {}'.format(step, sess.run(cost), sess.run(W), sess.run(b)))
        # cost가 줄어드는 것을 볼 수 있음


# 마지막에 똑같은 결과가 계속해서 나오는 이유는
# 두 번째 열에 출력된 cost가 0.0이 되어서 더 이상 비용을 줄일 수 없기 때문에 계산이 필요 없어서
