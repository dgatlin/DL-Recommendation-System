import pandas as pd
from pathlib import Path

from config.core import (
    DATASET_DIR,
    config,
)


# Load data set
def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    # rename variables beginning with numbers to avoid syntax errors later
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    return transformed


# Load data
file_path = config.app_config.training_data_file


def load_nums():
    # Get number of users and items
    counts = pd.read_csv(file_path)
    users = len(counts.iloc[4].unique())
    items = len(counts.iloc[5].unique())
    return [users, items]
