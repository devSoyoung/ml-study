import tensorflow as tf
import numpy as np

## 1. 데이터 준비
# [꼬리, 짖음, 다리개수, 크기]
x_data = np.array([[1, 1, 4, 50], [1, 0, 4, 5], [1, 0, 4, 20], [1, 0, 4, 10], [1, 1, 4, 100], [1, 0, 4, 12]])

# [dog, rat, cat] - 3종류
y_data = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
])


## 2. 신경망 설계
# X가 6가지 => W, b, L 6가지
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W1 = tf.Variable(tf.random_uniform([4, 4], -1., 1.))
b1 = tf.Variable(tf.zeros([4]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))       # matmul: 행렬 곱셈
# L1 해석: X와 W1을 곱하고, B1을 더하라 (y = X*W1 + b1)

W2 = tf.Variable(tf.random_normal([4, 5]))
b2 = tf.Variable(tf.zeros([5]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))      # ReLU: Rectified Linear Unit

# sigmoid: logistic classification에서 어디에 속하는지 분류를 하기 위해 사용
# 일정 값을 넘어야 성공/참이 될 수 있기 때문에, Activation Function이라고도 불림
#

W3 = tf.Variable(tf.random_normal([5, 6]))
b3 = tf.Variable(tf.zeros([6]))
L3 = tf.nn.relu(tf.add(tf.matmul))
