from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer


def build_preprocessing_pipeline(config):
    categorical_cols = config['features']['categorical']
    numeric_cols = ['recency', 'history', 'used_discount', 'used_bogo',
                    'history_discount', 'history_bogo', 'recency_history']

    kbins = KBinsDiscretizer(
        n_bins=config['features']['recency_bins'],
        encode='onehot-dense',
        strategy=config['features']['recency_bin_strategy']
    )

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('bin', kbins, ['recency'])
    ])

    return preprocessor