import os
import random
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.data_ingestion_config = config["data_ingestion"]
        self.bucket_name = self.data_ingestion_config["bucket_name"]
        self.object_name = self.data_ingestion_config["object_name"]
        self.storage_path = self.data_ingestion_config["storage_path"]
        self.extra_part = self.data_ingestion_config["extra_part"]

        self.url = f"https://{self.storage_path}/{self.bucket_name}/{self.object_name}?{self.extra_part}"

        artifact_dir = Path(self.data_ingestion_config["artfact_dir"])
        self.raw_dir = artifact_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_raw_data(self):
        raw_data_file = f"{self.raw_dir}/{self.object_name}"

        if os.path.exists(raw_data_file):
            logger.info(f"File already exists at {raw_data_file}, skipping download.")
            return raw_data_file

        retries = 3
        delay = 5

        for attempt in range(retries):
            try:
                logger.info(f"Attempt {attempt + 1} to connect to {self.url}")
                with requests.get(
                    self.url, headers={"User": "Mozilla/5.0"}, stream=True, timeout=100
                ) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("content-length", 0))
                    block_size = 1024
                    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

                    with open(raw_data_file, "wb") as f:
                        for data in r.iter_content(block_size):
                            progress_bar.update(len(data))
                            f.write(data)
                    progress_bar.close()

                    if total_size != 0 and progress_bar.n != total_size:
                        logger.error(
                            "ERROR: Downloaded file size does not match expected size."
                        )
                        raise Exception("Incomplete download")

                logger.info("Download completed successfully!")
                return raw_data_file

            except Exception as e:
                logger.error(f"Error: {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise e

    def save_to_csv_files(self, raw_data_file):
        df = pd.read_parquet(raw_data_file)
        raw_data_csv = f"{self.raw_dir}/{self.object_name.split('.')[0]}.csv"
        df.to_csv(raw_data_csv, index=False)
        logger.info(f"Raw data saved at {raw_data_csv}")

    def run(self):
        logger.info(f"Data ingestion started for f{self.url}")
        raw_data_file = self.download_raw_data()
        self.save_to_csv_files(raw_data_file)
        logger.info("Data ingestion complated successfully")
