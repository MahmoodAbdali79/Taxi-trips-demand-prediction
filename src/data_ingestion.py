import random
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.data_ingestion_config = config["data_ingestion"]
        self.bucket_name = self.data_ingestion_config["bucket_name"]
        self.object_name = self.data_ingestion_config["object_name"]
        self.storage_path = self.data_ingestion_config["storage_path"]
        self.extra_part = self.data_ingestion_config["extra_part"]
        self.train_ratio = self.data_ingestion_config["train_ratio"]

        self.url = f"https://{self.storage_path}/{self.bucket_name}/{self.object_name}?{self.extra_part}"

        artifact_dir = Path(self.data_ingestion_config["artfact_dir"])
        self.raw_dir = artifact_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_raw_data(self):
        req = Request(self.url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as response:
            raw_data = response.read().decode("utf-8")
            return raw_data

    def split_data(self, raw_data):
        numbers = [int(x) for x in raw_data.strip().split()]

        total_period, n_cols, n_rows = numbers[:3]
        data = numbers[3:]

        logger.info(f"Data format: {total_period} periods, {n_cols}x{n_rows} grid")

        test_data = []
        train_val_data = []

        for t in range(total_period):
            offest = t * n_cols * n_rows
            for row in range(n_rows):
                for col in range(n_cols):
                    demand = data[offest + row * n_cols + col]
                    if demand == -1:
                        test_data.append([t, row, col, demand])
                    else:
                        train_val_data.append([t, row, col, demand])

        random.shuffle(train_val_data)
        train_size = int(len(train_val_data) * self.train_ratio)

        train_data = train_val_data[:train_size]
        val_data = train_val_data[train_size:]

        return train_data, val_data, test_data

    def save_to_csv_files(self, train_data, val_data, test_data):

        header = "time,row,col,demand\n"

        data_file = [
            ("train", train_data),
            ("validation", val_data),
            ("test", test_data),
        ]

        for name, data in data_file:
            output_file = self.raw_dir / f"{name}.csv"
            with open(output_file, "w") as f:
                f.write(header)
                for row in data:
                    f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
                logger.info(f"Saved {name} data to {output_file}")

    def run(self):
        logger.info(f"Data ingestion started for f{self.url}")
        raw_data = self.download_raw_data()
        train_data, val_data, test_data = self.split_data(raw_data)
        self.save_to_csv_files(train_data, val_data, test_data)
        logger.info("Data ingestion complated successfully")
