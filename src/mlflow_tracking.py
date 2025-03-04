import mlflow
import mlflow.sklearn

def start_mlflow_run():
    mlflow.start_run()

def end_mlflow_run():
    mlflow.end_run()

def log_param(key, value):
    mlflow.log_param(key, value)

def log_metric(key, value):
    mlflow.log_metric(key, value)

def log_model(model, model_name):
    mlflow.sklearn.log_model(model, model_name)
