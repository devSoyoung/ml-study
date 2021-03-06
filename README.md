
# 1. 개발환경 구축하기
## 아나콘다 설치
### 아나콘다
Continuum Analytics에서 제작한 파이썬 배포판. 다양한 파이썬 패키지를 포함하고 있으며, 상업용으로도 무료 사용이 가능. **패키지 의존성을 관리**해주기 때문에, 가상환경에 따라 독립적으로 패키지 관리가 가능하다는 장점이 있음.

>현재 아나콘다 홈페이지에는 최신 파이썬 버전인 python 3.7 버전의 설치 파일을 제공하고 있다. 다른 버전의 파이썬을 깔고 싶다면, 설치한 후에 다음 명령어를 실행하면 된다. [[공식홈페이지 링크]](http://docs.anaconda.com/anaconda/user-guide/faq/#how-do-i-get-the-latest-anaconda-with-python-3-5)

>python 3.7이 설치된 상태에서 virtual env를 생성한 후 tensorflow를 설치하려고 하니 알아서 python 버전을 내려주었다.

	conda install python={version}

텐서플로우는 패키지들 버전에 민감(버전이 다르면 안 돌아가는 예민한 친구;)하기 때문에 아나콘다로 가상환경을 만들어서 텐서플로우 전용 환경을 구축해주자.

## 새 가상환경 생성 (w. anaconda)

	conda create -n "{venv_name}" python=3 anaconda

anaconda를 같이 설치해주어야 jupyter notebook을 사용할 때, 가상환경의 파이썬을 사용할 수 있음

### 가상환경 삭제
	conda remote -n "{venv_name}" --all 

### 가상환경 활성화
	source activate {venv_name}

### 가상환경 비활성화
	source deactivate

### 가상환경 목록 확인
	conda info --envs

### 생성한 가상환경에 패키지 추가 설치
	conda install -n "{venv_name}" {packages..}
	
***
# 2. Hyperparameter
신경망 학습을 통해 최적화되는 변수(편향, 가중치)가 아니라 학습진도율, 일반화 변수처럼 경험적 지식으로 설정을 하거나 외부 모델 메커니즘을 통해 자동 설정되는 변수. meta parameter, free parameter라고도 한다.

* Learning rate
* Cost function
* Regularization parameter
* Mini batch size
* Training iteration count
* Hidden unit
* Weight initialization

어떤 값이 최적인지는 아직 밝혀진바가 없으며, 다양한 시도를 통해 적절한 값을 찾아야함.

* 참고링크 : [최적의 하이퍼파라미터 값 찾는 방법](https://m.blog.naver.com/laonple/220576650094)

***
# 3. Overfitting
데이터를 지나치게 학습해서 학습한 데이터에 대해서만 정답을 잘 도출하고, 다른 데이터를 넣으면 정답을 맞추지 못하는 현상.
* 참고링크 : [과적합 풀어서 보기](https://hyperdot.wordpress.com/2017/02/06/%EA%B3%BC%EC%A0%81%ED%95%A9overfitting/)

## 해결방법
1. 레이어에 dropout 적용
2. 너무 많이 학습하지 않는다..

***
# 4. One-hot encoding
표현하고자 하는 값의 인덱스만 1로 하고 나머지는 모두 0으로 하는 데이터 표현방식. 

## 장점
* 각각으로 분류되는 데이터를 가중치 없이 표현하기에 좋음

## 단점
* 표현하려는 데이터가 늘어날수록 벡터의 차원을 늘려야함
* 데이터간의 유사성을 표현할 수 없음 (검색 시스템에서 불리[[참고링크]](https://wikidocs.net/22647))

***
# 5. 단어 임베딩
의미를 최대한 담아서 단어를 벡터로 변환하는 방법. 하나의 단어가 미리 정의된 차원(보통 20~200)에서 연속형의 값을 갖는 벡터로 표현. 각각의 차원은 모두 정보를 가지고 있으므로, 벡터 연산을 통해 유사도를 측정할 수 있음
* 참고링크 : [단어간 유사도 파악 방법](http://www.kakaobrain.com/blog/6)

***
# Amazon SageMaker 자동 모델 튜닝 기능
최적의 하이퍼파라미터를 찾기 위해서 노가다를 하던 중 발견. 공부해서 적용해야겠다. 노가다 너무 힘들어..
* 참고링크 : [Amazon SageMaker 자동 모델 튜닝 기능 출시 - 기계학습을 위한 인공지능](https://aws.amazon.com/ko/blogs/korea/sagemaker-automatic-model-tuning/)

***
# *. 주요 텐서플로우 함수
* tf.argmax(): 텐서에서 제일 큰 값의 인덱스를 반환
* tf.truncated_normal(): 평균이 0에 가깝고, 값이 0에 가까운 정규분포에서 난수 선택
* tf.random_normal(): 평균이 0에 가까운 정규분포에서 난수 선택
* tf.keras.he_normal():

***
# *. 용어 정리
* batch : 한 번에 학습하는 데이터의 수
* label : y값, 결과값
* epoch : 학습 데이터 **전체**를 가지고 한 번의 학습을 완료했을 때, 한 세대를 의미.
* axis : 중심선
* optimizer : cost 함수의 최솟값을 찾는 알고리즘
