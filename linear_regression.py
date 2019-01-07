# X와 Y의 상관관계를 분석하는 기초적인 선형 회귀 모델

import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 이름을 붙여주는 이유: 텐서보드 등으로 값의 변화를 추적하거나 쉽게 살펴보려고
X = tf.placeholder(tf.float32, [1], name="X")
Y = tf.placeholder(tf.float32, [1], name="Y")
print(X)
print(Y)

# X, Y의 상관관계를 분석하기 위한 가설 수식
# y = W * X + b
hypothesis = W * X + b

# 손실 함수 작성
# mean(): 예측값과 실제값의 거리를 비용 함수로 지정
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 경사하강법 최적화 수행(텐서플로우 내장 함수 이용)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 비용 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

# 세션 생성 및 초기화
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화 100번 수행
    for step in range(100):
        # sess.run으로 train_op와 cost 그래프 계산
        # 가설 수식에 넣어야 할 실제값을 feed_dict를 통해 전달
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data,Y:y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n===Test===")
    print("X:5, Y:", sess.run(hypothesis, feed_dict={X:5}))
    print("X:2.5, Y:", sess.run(hypothesis), feed_dict={X:2.5})