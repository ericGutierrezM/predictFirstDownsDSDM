import pandas as pd
import numpy as np

from src.firstDown.feature_engineering.encode import (
    one_hot,
    one_hot_transform
)

def test_one_hot_returns_encoder():
    df = pd.DataFrame({
        'team': ['A', 'B', 'A']
    })

    encoder = one_hot(df, ['team'])

    assert encoder is not None
    assert hasattr(encoder, 'categories_')

def test_one_hot_transform_creates_columns():
    df = pd.DataFrame({
        'team': ['A', 'B', 'A']
    })

    encoder = one_hot(df, ['team'])
    result = one_hot_transform(df, ['team'], encoder)

    assert 'team' not in result.columns

    assert any(col.startswith('team_') for col in result.columns)

def test_one_hot_transform_preserves_row_count():
    df = pd.DataFrame({
        'team': ['A', 'B', 'A']
    })

    encoder = one_hot(df, ['team'])
    result = one_hot_transform(df, ['team'], encoder)

    assert len(result) == len(df)
