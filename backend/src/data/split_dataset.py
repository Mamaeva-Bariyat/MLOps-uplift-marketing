from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


def split_data(df, config):
    stratify_key = df['treatment'].astype(str) + df['conversion'].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=config['train']['test_size'],
        random_state=config['train']['random_state'],
        stratify=stratify_key
    )

    processed_path = Path(config['data']['processed_path'])
    train_df.to_csv(processed_path / config['data']['train_file'], index=False)
    test_df.to_csv(processed_path / config['data']['test_file'], index=False)

    return train_df, test_df