import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    mlflow.start_run()

    # Load the preprocessed data
    X_test = load_data("data/X_test.csv")
    y_test = load_data("data/y_test.csv")

    # Load the trained model
    model = mlflow.sklearn.load_model("model")

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    print(f"Model precision: {precision}")
    print(f"Model recall: {recall}")
    print(f"Model F1 score: {f1}")

    # Log the metrics to MLFlow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.end_run()
