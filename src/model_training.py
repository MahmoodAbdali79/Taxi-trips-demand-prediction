from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from src.logger import get_logger
from src.model_factory import get_model

logger = get_logger(__name__)


class ModelTraining:
    """
    Handles the training, evaluation, and saving of a Random Forest model
    for predicting taxi trip demand. Also integrates with MLflow for experiment tracking.

    Attributes:
        processed_dir (Path): Path to directory containing processed training/validation data.
        model_output_dir (Path): Path to save the trained model.
        model_training_config (dict): Hyperparameters and configuration for training.
    """

    def __init__(self, config):
        """
        Initializes the ModelTraining class with the given configuration.

        Args:
            config (dict): Configuration dictionary containing training and data paths.
        """
        artifact_dir = Path(config["data_ingestion"]["artfact_dir"])
        self.processed_dir = artifact_dir / "processed"
        self.model_training_config = config["traning_model"]
        self.model_output_dir = artifact_dir / "models"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """
        Loads the preprocessed training and validation datasets.

        Returns:
            tuple: A tuple of (train_data, val_data), both as pandas DataFrames.
        """
        self.train_path = self.processed_dir / "train.csv"
        self.val_path = self.processed_dir / "val.csv"
        train_data = pd.read_csv(self.train_path)
        val_data = pd.read_csv(self.val_path)
        return train_data, val_data

    def build_model(self):
        """
        Builds and returns a RandomForestRegressor using the specified hyperparameters.

        Returns:
            RandomForestRegressor: The initialized random forest model.
        """
        model_name = self.model_training_config["selected_model"]
        selected_model = next(
            (
                m
                for m in self.model_training_config["models_to_run"]
                if m["name"] == model_name
            ),
            None,
        )
        if selected_model is None:
            logger.error(f"Model '{model_name}' not found in config.yaml")
            raise ValueError(f"Model '{model_name}' not found in config.yaml")
        model = get_model(selected_model["name"], selected_model["params"])

        return model

    def train_model(self, model, train_data):
        """
        Trains the given model on the training dataset.

        Args:
            model (RandomForestRegressor): The model to be trained.
            train_data (pd.DataFrame): Training dataset.

        Returns:
            RandomForestRegressor: The trained model.
        """
        self.model_output_dir
        X_train, y_train = (
            train_data.drop(columns=["trip_count"]),
            train_data["trip_count"],
        )
        model.fit(X_train, y_train)
        return model

    def evaluation_model(self, model, val_data):
        """
        Evaluates the trained model on the validation dataset.

        Logs root mean squared error (RMSE) and out-of-bag score.

        Args:
            model (RandomForestRegressor): The trained model.
            val_data (pd.DataFrame): Validation dataset.
        """
        X_val, y_val = val_data.drop(columns=["trip_count"]), val_data["trip_count"]
        y_pred = model.predict(X_val)
        y_pred = [round(x) for x in y_pred]
        self.rmse = root_mean_squared_error(y_val, y_pred)
        self.oob_score = model.oob_score_
        logger.info(f"Out-of-bag score: {self.oob_score}")
        logger.info(f"Root mean squred error is {self.rmse}")

    def save(self, model):
        """
        Saves the trained model to disk in joblib format.

        Args:
            model (RandomForestRegressor): The trained model.
        """
        self.model_output_path = self.model_output_dir / "rf.joblib"
        joblib.dump(model, self.model_output_path, compress=("lzma", 5))
        logger.info(f"Model saved at {self.model_output_dir}")

    def run(self):
        """
        Executes the end-to-end training pipeline:
        - Loads data
        - Builds and trains the model
        - Evaluates the model
        - Logs results and parameters using MLflow
        - Saves the model to disk
        """
        mlflow.set_experiment("Taxi_demand")
        with mlflow.start_run():
            logger.info("Training model started")
            logger.info("MLflow started")
            mlflow.set_tag("model_type", "random_forest")

            train_data, val_data = self.load_data()
            mlflow.log_artifact(self.train_path, "datasets")
            mlflow.log_artifact(self.val_path, "datasets")

            model = self.build_model()
            self.train_model(model, train_data)
            self.evaluation_model(model, val_data)
            mlflow.log_metric("rmse", self.rmse)
            mlflow.log_metric("obb_score", self.oob_score)

            self.save(model)
            mlflow.log_artifact(self.model_output_path, "models")
            params = model.get_params()
            mlflow.log_params(params)
            logger.info("MLflow complated successfully")
            logger.info("Model training complated successfully")
