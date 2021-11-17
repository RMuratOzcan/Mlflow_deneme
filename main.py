import mlflow
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
df_wine = pd.read_csv('winequality-red.csv',sep= ";")
X = df_wine.drop(columns = 'quality')
y = df_wine[['quality']]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42)
alpha = 1
l1_ratio = 1
with mlflow.start_run():
    model = ElasticNet(alpha = alpha,
                       l1_ratio = l1_ratio)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(preds, y_val)
    abs_error = mean_absolute_error(preds, y_val)
    r2 = r2_score(preds, y_val)
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('l1_ratio', l1_ratio)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('abs_error', abs_error)
    mlflow.log_metric('r2', r2)
    mlflow.sklearn.log_model(model, 'model')
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.register_model("sqlite:///mlruns.db",
                      "pycharm_veri")