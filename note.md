## ML-FLOW 

### Azure 가상 머신으로 진행 

### Ubuntu 20,04 LTS 머신 생성

###
2. MLFlow 링크
ML의 라이프 사이클을 관리
ML 모델의 실험을 tracking하고 model을 공유 및 deploy 할 수 있도록 지원하는 라이브러리
a. 제공하는 기능
Model tracking : ML 모델을 학습시킬 때 생기는 각종 파라미터, metric 결과 등을 logging하고 그 결과를 web ui에서 확인
MLflow Projects : anaconda, docker 등을 사용해서 만들어 둔 모델을 reproducible하고 실행할 수 있도록 코드 패키지 형태로 제공
MLflow Models : 동일한 모델을 Docker, Spark, AWS 등에 쉽게 배치할 수 있도록 지원
MLflow Model Registry : MLflow 모델의 전체 라이프사이클을 공동으로 관리하기 위한 기능 제공
b. 장단점
장점
MLFlow는 실험 및 버전 관리 모델을 추적하기 위한 Python 프로그램
즉 인프라에 제한 없이 어느 환경에서나 실행할 수 있고, 단일 서비스로 운영되어 사용하기가 쉬움
단점
ML 모델 트레킹에 중점된 기능을 제공하기 때문에 Kubeflow와 같이 전체 AI 플랫폼으로 사용되기에 어려움이 있음(별도의 오픈소스와 함께 사용할 필요가 있음)
3. 오픈소스 결정
Kubeflow의 경우 쿠버네티스 환경을 구축과 운영 상이 리소스 관리 측면에서 얼려움이 있음
추가로 프로젝트 비용을 고려했을 때 충분한 리소스 활용이 어려울 것으로 판단해 Kubeflow를 사용하기 보단 MLFlow를 기본으로 사용하고 필요한 기술을 붙여서 사용하는 방법으로 진행
MLFlow 기반의 모델 tracking 오픈소스 사용



```
# python 설치
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

## package 업데이트 되었는지 확인
apt list | grep python3.10

sudo apt install python3.10

## 설치 확인
python3 --version

## 경로 지정
mkdir ~/mlflow && cd ~/mlflow

## 가상환경 설치
sudo apt install python3.8-venv
python3 -m venv .venv

## 가상환경 실행
source .venv/bin/activate

# mlflow 설치
pip3 install mlflow

# mlflow 실행
## host 지정해야만 외부에 접속할 수 있는 IP 열림
mlflow ui --host 0.0.0.0
```

Azure에서 인바운드 규칙 5000번 열여주기!!


```
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

experiment_name = "minjun_researcher"
run_name = "20240116-sklearn"

mlflow.set_tracking_uri("{url}:5000")
mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()

with mlflow.start_run(run_name=run_name) as run:
    noise = np.random.rand(100, 1)
    X = sorted(10 * np.random.rand(100, 1)) + noise
    y = sorted(10 * np.random.rand(100))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(preds)

```