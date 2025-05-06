from config_reader import read_config
from data_ingestion import DataIngestion
from data_processing import DataProcessing
from logger import get_logger
from model_training import ModelTraining

# logger = get_logger(__name__)
config = read_config("config/config.yaml")


data_ingestion = DataIngestion(config)
data_ingestion.run()

# data_precessing = DataProcessing(config)
# data_precessing.run()

# model_training = ModelTraining(config)
# model_training.run()
