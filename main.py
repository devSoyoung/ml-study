from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f'.format

# 데이터 세트 로드
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# 데이터 무작위 추출
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0

# 모델 만들기
# 1) 특성 정의, 특성 열 구성
my_feature = california_housing_dataframe[["total_rooms"]]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# 2) 타겟 정의
targets = california_housing_dataframe["median_house_value"]

# 3) LinearRegressor 구성
# 모델을 학습시키기 위해 GradientDescentOptimizer를 사용
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 선형 회기 모델 구성, 학습률 : 0.0000001로 지정(optimizer)
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


# 4) 입력 함수 정의
def my_input_fun(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 5) 모델 학습, 100단계 학습하도록 설정
_ = linear_regressor.train(
    input_fn=lambda: my_input_fun(my_feature, targets),
    steps=100
)

# 6) 모델 평가 - 값 예측 후 오차 계산
# MSE(평균제곱오차)와 RMSE(평균제곱근오차) 계산
prediction_input_fn = lambda: my_input_fun(my_feature, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([item['predictions'][0] for item in predictions])

mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data: %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on ..) : %0.3f" % root_mean_squared_error)

# RMSE를 타겟의 최소, 최댓값의 차와 비교
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Diff between Min and Max: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)


# 모델 오차 줄이기
# 전반적 요약 통계를 참조해 예측과 타겟이 일치율 조사
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()

# 무작위 데이터 샘플 추출, 300개
sample = california_housing_dataframe.sample(n=300)

# 최소, 최댓값 가져오기
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# 가중치, 편향 가져오기
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias
