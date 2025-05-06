from pathlib import Path

import yaml

# from logger import get_logger
from src.logger import get_logger

logger = get_logger(__name__)


def read_config(config_path):
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str or Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If the file exists but contains invalid YAML syntax.

    Example:
        config = read_config("config/config.yaml")
        print(config['data_ingestion']['bucket_name'])
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.error(f"config file not found at {config_path}")
        raise FileNotFoundError(f"config file not found at {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        logger.error(f"error parsing yaml file {config_path}: {e}")
        raise


if __name__ == "__main__":
    config = read_config("config/config.yaml")
    # print(config['data_ingestion'])

    print(config["data_ingestion"]["bucket_name"])
