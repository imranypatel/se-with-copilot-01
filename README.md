# End-to-End Machine Learning Project with MLOps using MLFlow

This project demonstrates an end-to-end machine learning workflow with MLOps using MLFlow. The project includes data preprocessing, model training, model evaluation, and MLFlow tracking for the complete lifecycle of the machine learning project.

## Project Structure

- `src/`: Source code for the project
  - `data_preprocessing.py`: Code for data preprocessing using the `DataPreprocessor` class
  - `model_training.py`: Code for model training using the `ModelTrainer` class
  - `model_evaluation.py`: Code for model evaluation using the `ModelEvaluator` class
  - `mlflow_tracking.py`: Code for MLFlow tracking using the `MLFlowTracker` class
- `requirements.txt`: Dependencies for the project

## Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/imranypatel/se-with-copilot-01.git
   cd se-with-copilot-01
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. Run the data preprocessing script:
   ```bash
   python src/data_preprocessing.py
   ```

2. Run the model training script:
   ```bash
   python src/model_training.py
   ```

3. Run the model evaluation script:
   ```bash
   python src/model_evaluation.py
   ```

## Using MLFlow

1. Start the MLFlow server:
   ```bash
   mlflow ui
   ```

2. Open the MLFlow UI in your browser:
   ```
   http://localhost:5000
   ```

3. Track experiments and models using the MLFlow UI.
