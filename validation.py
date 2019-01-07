# * 라벨(타겟, 결과) : median_house_value

from __future__ import print_function

import math

# from IPython import display
# from matplotlib import cm
# from matplotlib import gridspec
from matplotlib import pyplot as plt        # 그래프 그리기 관련
import numpy as np
import pandas as pd     # 데이터 프레임 관련 라이브러리
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# 데이터 세트 로드
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# 입력 함수 정의
def my_input_fun(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key,value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 특성 열 구성
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


# 모델 구축
def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    training_input_fn = lambda: my_input_fun(
        training_examples,
        training_targets["median_house_value"],
        batch_size=batch_size
    )
    predict_training_input_fn = lambda: my_input_fun(
        training_examples,
        training_targets["median_house_value"],
        num_epochs=1,
        shuffle=False
    )
    predict_validation_input_fn = lambda: my_input_fun(
        validation_examples, validation_targets["median_house_value"],
        num_epochs=1,
        shuffle=False
    )

    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range (0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )

        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, validation_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        print("period %02d : %0.2f" % (period, training_root_mean_squared_error))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # 결과 출력하기
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor

# 여러 특성을 사용하기 때문에, 전처리 필요
def preprocess_features(california_housing_dataframe):
    # 8개 특징을 선택하고, 1개는 추가적으로 만들어서(rooms_per_person) 넣어줌
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] /
        california_housing_dataframe["population"])
    return processed_features


def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets["median_house_value"] = (
        california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


# 학습 세트 선정 - 12,000개
training_examples = preprocess_features(california_housing_dataframe.head(12000))
# training_examples.describe()
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
# training_targets.describe()

# 검증 세트 선정 - 5,000개
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

linear_regressor = train_model(
    learning_rate=0.00001,
    steps=100,
    batch_size=1,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)

# # 그래프 그리기
# plt.figure(figsize=(13, 8))

# # 검증 데이터 그래프 그리기
# ax = plt.subplot(1, 2, 1)
# ax.set_title("Validation Data")
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(validation_examples["longitude"],
#             validation_examples["latitude"],
#             cmap="coolwarm",
#             c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
#
# # 학습 데이터 그래프 그리기
# ax = plt.subplot(1, 2, 2)
# ax.set_title("Training Data")
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(training_examples["longitude"],
#             training_examples["latitude"],
#             cmap="coolwarm",
#             c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
# _ = plt.show()
