from pathlib import Path

import pandas as pd

from src.config_reader import read_config
from src.logger import get_logger

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, config):
        self.data_processing_config = config["data_processing"]

        artifacts_dir = Path(config["data_ingestion"]["artfact_dir"])
        self.row_dir = artifacts_dir / "raw"
        self.processed_dir = artifacts_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_row_data(self):
        train_data = pd.read_csv(self.row_dir / "train.csv")
        val_data = pd.read_csv(self.row_dir / "validation.csv")
        test_data = pd.read_csv(self.row_dir / "test.csv")
        return train_data, val_data, test_data

    def process_data(self, train_data, val_data, test_data):
        processed_train_data = self._precess_single_dataset(train_data)
        processed_val_data = self._precess_single_dataset(val_data)
        processed_test_data = self._precess_single_dataset(test_data)
        return processed_train_data, processed_val_data, processed_test_data

    def save_to_csv_file(
        self, processed_train_data, processed_val_data, processed_test_data
    ):
        column_order = ["hour_of_day", "day", "row", "col", "demand"]

        train_data = processed_train_data[column_order]
        val_data = processed_val_data[column_order]
        test_data = processed_test_data[column_order]

        train_data.to_csv(self.processed_dir / "train.csv", index=False)
        val_data.to_csv(self.processed_dir / "val.csv", index=False)
        test_data.to_csv(self.processed_dir / "test.csv", index=False)
        logger.info(f"Saved processed data to {self.processed_dir}")

    def _precess_single_dataset(self, data):
        data = data.sort_values(by=["time", "row", "col"]).reset_index(drop=True)
        data["time"] = data["time"] + self.data_processing_config["shift"]
        data = data.assign(hour_of_day=data["time"] % 24, day=data["time"] // 24)
        data = data.drop(columns=["time"])
        # print(data.shape)
        return data

    def run(self):
        logger.info("Data processing started")
        train_data, val_data, test_data = self.load_row_data()
        precessed_train_data, precessed_val_data, precessed_test_data = (
            self.process_data(train_data, val_data, test_data)
        )
        self.save_to_csv_file(
            precessed_train_data, precessed_val_data, precessed_test_data
        )
        logger.info("Data processing complated successfully")
