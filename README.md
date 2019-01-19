
# 1. 개발환경 구축하기
## 아나콘다 설치
### 아나콘다
Continuum Analytics에서 제작한 파이썬 배포판. 다양한 파이썬 패키지를 포함하고 있으며, 상업용으로도 무료 사용이 가능. **패키지 의존성을 관리**해주기 때문에, 가상환경에 따라 독립적으로 패키지 관리가 가능하다는 장점이 있음.

>현재 아나콘다 홈페이지에는 최신 파이썬 버전인 python 3.7 버전의 설치 파일을 제공하고 있다. 다른 버전의 파이썬을 깔고 싶다면, 설치한 후에 다음 명령어를 실행하면 된다. [[공식홈페이지 링크]](http://docs.anaconda.com/anaconda/user-guide/faq/#how-do-i-get-the-latest-anaconda-with-python-3-5)

>python 3.7이 설치된 상태에서 virtual env를 생성한 후 tensorflow를 설치하려고 하니 알아서 python 버전을 내려주었다.

	conda install python={version}
	


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
# *. 주요 텐서플로우 함수i
* tf.argmax(): 텐서에서 제일 큰 값의 인덱스를 반환
* tf.truncated_normal(): 평균이 0에 가깝고, 값이 0에 가까운 정규분포에서 난수 선택
* tf.random_normal(): 평균이 0에 가까운 정규분포에서 난수 선택

***
# *. 용어 정리
* batch : 한 번에 처리하는 데이터의 수
* label : y값, 결과값
* epoch : 학습 데이터 전체를 가지고 한 번의 학습을 완료했을 때, 한 세대를 의미.
* axis : 중심선
* optimizer : cost 함수의 최솟값을 찾는 알고리즘
