import pandas as pd
from pathlib import Path

def load_raw_data(config):
    raw_path = Path(config['data']['raw_path'])
    return pd.read_csv(raw_path)