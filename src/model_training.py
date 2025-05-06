from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from src.logger import get_logger

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, config):
        artifact_dir = Path(config["data_ingestion"]["artfact_dir"])
        self.processed_dir = artifact_dir / "processed"
        self.model_training_config = config["traning_model"]
        self.model_output_dir = artifact_dir / "models"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.train_path = self.processed_dir / "train.csv"
        self.val_path = self.processed_dir / "val.csv"
        train_data = pd.read_csv(self.train_path)
        val_data = pd.read_csv(self.val_path)
        return train_data, val_data

    def build_model(self):
        n_estimators = self.model_training_config["n_estimators"]
        max_samples = self.model_training_config["max_samples"]
        n_jobs = self.model_training_config["n_jobs"]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=n_jobs,
            oob_score=root_mean_squared_error,
        )
        return model

    def train_model(self, model, train_data):
        self.model_output_dir
        X_train, y_train = train_data.drop(columns=["demand"]), train_data["demand"]
        model.fit(X_train, y_train)
        return model

    def evaluation_model(self, model, val_data):
        X_val, y_val = val_data.drop(columns=["demand"]), val_data["demand"]
        y_pred = model.predict(X_val)
        y_pred = [round(x) for x in y_pred]
        self.rmse = root_mean_squared_error(y_val, y_pred)
        self.oob_score = model.oob_score_
        logger.info(f"Out-of-bag score: {self.oob_score}")
        logger.info(f"Root mean squred error is {self.rmse}")

    def save(self, model):
        self.model_output_path = self.model_output_dir / "rf.joblib"
        joblib.dump(model, self.model_output_path, compress=("lzma", 3))
        logger.info(f"Model saved at {self.model_output_dir}")

    def run(self):
        mlflow.set_experiment("tap40")
        with mlflow.start_run():
            logger.info("Training model started")
            logger.info("MLflow started")
            mlflow.set_tag("model_type", "ranfon_forest")

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
