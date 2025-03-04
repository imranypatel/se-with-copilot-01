import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        self.data = self.data.dropna()  # Remove missing values
        self.data = pd.get_dummies(self.data)  # Convert categorical variables to dummy variables

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_data(self):
        self.X_train.to_csv("data/X_train.csv", index=False)
        self.X_test.to_csv("data/X_test.csv", index=False)
        self.y_train.to_csv("data/y_train.csv", index=False)
        self.y_test.to_csv("data/y_test.csv", index=False)

if __name__ == "__main__":
    file_path = "data/dataset.csv"
    target_column = "target"

    preprocessor = DataPreprocessor(file_path, target_column)
    preprocessor.load_data()
    preprocessor.preprocess_data()
    preprocessor.split_data()
    preprocessor.save_data()
