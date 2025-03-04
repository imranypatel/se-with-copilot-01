import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self):
        self.model = None

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def train_model(self, X_train, y_train):
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

if __name__ == "__main__":
    mlflow.start_run()

    trainer = ModelTrainer()

    # Load the preprocessed data
    X_train = trainer.load_data("data/X_train.csv")
    X_test = trainer.load_data("data/X_test.csv")
    y_train = trainer.load_data("data/y_train.csv")
    y_test = trainer.load_data("data/y_test.csv")

    # Train the model
    trainer.train_model(X_train, y_train)

    # Evaluate the model
    accuracy = trainer.evaluate_model(X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    # Log the model and metrics to MLFlow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(trainer.model, "model")

    mlflow.end_run()
