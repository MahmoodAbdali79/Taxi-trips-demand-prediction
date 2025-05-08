from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config_reader import read_config
from src.logger import get_logger

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, config):
        self.data_processing_config = config["data_processing"]
        self.data_ingestion_config = config["data_ingestion"]

        self.val_test_ratio = self.data_processing_config["val_test_ratio"]
        self.test_ratio = self.data_processing_config["test_ratio"]

        artifacts_dir = Path(self.data_ingestion_config["artfact_dir"])
        self.row_dir = artifacts_dir / "raw"
        self.object_name = self.data_ingestion_config["object_name"]
        self.processed_dir = artifacts_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_row_data(self):
        raw_data = pd.read_csv(f"{self.row_dir}/{self.object_name.split('.')[0]}.csv")
        return raw_data

    def process_data(self, df):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df.dropna(subset=["PUlocationID"], inplace=True)
        df["PUlocationID"] = df["PUlocationID"].astype(int)
        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_day"] = df["pickup_datetime"].dt.day
        df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
        df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype(int)
        df = (
            df.groupby(
                [
                    "pickup_day",
                    "pickup_hour",
                    "pickup_weekday",
                    "is_weekend",
                    "PUlocationID",
                ]
            )
            .size()
            .reset_index(name="trip_count")
        )
        logger.info(f"Data proccesed. shape of data: {df.shape}")
        return df

    def split_data(self, df):
        df_shuffle = df.sample(frac=1, random_state=4).reset_index(drop=True)
        train_df, df_temp = train_test_split(
            df_shuffle, test_size=self.val_test_ratio, random_state=42
        )
        val_df, test_df = train_test_split(
            df_temp, test_size=self.val_test_ratio, random_state=42
        )
        return train_df, val_df, test_df

    def save_to_csv_file(self, train_data, val_data, test_data):
        train_data.to_csv(self.processed_dir / "train.csv", index=False)
        val_data.to_csv(self.processed_dir / "val.csv", index=False)
        test_data.to_csv(self.processed_dir / "test.csv", index=False)
        logger.info(f"Saved processed data to {self.processed_dir}")

    def run(self):
        logger.info("Data processing started")
        raw_data = self.load_row_data()
        precessed_data = self.process_data(raw_data)
        train_data, val_data, test_data = self.split_data(precessed_data)
        self.save_to_csv_file(train_data, val_data, test_data)
        logger.info("Data processing complated successfully")
