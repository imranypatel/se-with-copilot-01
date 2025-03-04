import mlflow
import mlflow.sklearn

class MLFlowTracker:
    def __init__(self):
        pass

    def start_mlflow_run(self):
        mlflow.start_run()

    def end_mlflow_run(self):
        mlflow.end_run()

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_model(self, model, model_name):
        mlflow.sklearn.log_model(model, model_name)
