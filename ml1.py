import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

experiment_name = "chanyoung"
run_name = "mlflow-example"

mlflow.set_tracking_uri("http://20.214.136.28:5000")
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
