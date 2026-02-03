import pandas as pd
from pathlib import Path
import yaml

def load_config(project_root: Path):
    """Load configuration from params.yml."""
    config_path = project_root / 'config' / 'params2.yml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_test_data(project_root: Path):
    """Load test data from processed path in config."""
    config = load_config(project_root)
    test_path = project_root / config['data']['processed_path'] / config['data']['test_file']
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    return pd.read_csv(test_path)