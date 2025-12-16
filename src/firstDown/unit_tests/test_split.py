import pandas as pd
from sklearn.model_selection import train_test_split

from src.firstDown.preprocessing.split import split_data

def test_split_data_basic():
    df = pd.DataFrame({
        'feature1': range(10),
        'feature2': range(10, 20),
        'target': [0,1,0,1,0,1,0,1,0,1]
    })

    X_train, X_test, y_train, y_test = split_data(df, 'target', test_size=0.3)

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3

    assert 'target' not in X_train.columns
    assert 'target' not in X_test.columns

    X_train2, X_test2, y_train2, y_test2 = split_data(df, 'target', test_size=0.3)
    pd.testing.assert_frame_equal(X_train, X_train2)
    pd.testing.assert_frame_equal(X_test, X_test2)
    pd.testing.assert_series_equal(y_train.reset_index(drop=True), y_train2.reset_index(drop=True))
    pd.testing.assert_series_equal(y_test.reset_index(drop=True), y_test2.reset_index(drop=True))

def test_split_data_half():
    df = pd.DataFrame({
        'feature': range(8),
        'target': [1,0,1,0,1,0,1,0]
    })

    X_train, X_test, y_train, y_test = split_data(df, 'target', test_size=0.5)
    assert len(X_train) == 4
    assert len(X_test) == 4

def test_split_data_half():
    df = pd.DataFrame({
        'feature': range(8),
        'target': [1,0,1,0,1,0,1,0]
    })

    X_train, X_test, y_train, y_test = split_data(df, 'target', test_size=0.5)
    assert len(X_train) == 4
    assert len(X_test) == 4

def test_split_data_single_feature():
    df = pd.DataFrame({
        'feature': range(5),
        'target': [1,0,1,0,1]
    })

    X_train, X_test, y_train, y_test = split_data(df, 'target', test_size=0.4)
    assert 'feature' in X_train.columns
    assert 'target' not in X_train.columns
