import pandas as pd
from pathlib import Path

from config.core import (
    DATASET_DIR,
    TRAINED_MODEL_DIR,
    config,
)


# Load data set
def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    # rename variables beginning with numbers to avoid syntax errors later
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    return transformed
