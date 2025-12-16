import pandas as pd
import numpy as np

from src.firstDown.preprocessing.deal_nan import (
    search_nan,
    replace_nan,
    drop_nan
)

def test_search_nan_counts_correctly():
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [np.nan, np.nan, 3]
    })

    result = search_nan(df)

    assert 'var' in result.columns
    assert 'rows_nan' in result.columns
    assert 'rows_other' in result.columns
    assert 'rows_total' in result.columns

    assert result.loc[result['var']=='A', 'rows_nan'].iloc[0] == 1
    assert result.loc[result['var']=='B', 'rows_nan'].iloc[0] == 2

def test_replace_nan_mean():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    result, value = replace_nan(df.copy(), 'A', 'mean')
    expected_mean = (1 + 3) / 2
    assert (result['A'].iloc[1] == expected_mean)
    assert value == expected_mean

def test_replace_nan_median():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    result, value = replace_nan(df.copy(), 'A', 'median')
    assert (result['A'].iloc[1] == 2)
    assert value == 2

def test_replace_nan_min_max_num_text():
    df = pd.DataFrame({'A': [1, np.nan, 3]})
    
    # min
    result, val = replace_nan(df.copy(), 'A', 'min')
    assert result['A'].iloc[1] == 1
    assert val == 1
    
    # max
    result, val = replace_nan(df.copy(), 'A', 'max')
    assert result['A'].iloc[1] == 3
    assert val == 3

    # num
    result, val = replace_nan(df.copy(), 'A', 'num', num=99)
    assert result['A'].iloc[1] == 99
    assert val == 99

    # text
    df_text = pd.DataFrame({'A': ['x', None, 'y']})
    result, val = replace_nan(df_text.copy(), 'A', 'text', txt='missing')
    assert result['A'].iloc[1] == 'missing'
    assert val == 'missing'

def test_drop_nan_removes_rows():
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })

    result = drop_nan(df, ['A'])
    assert len(result) == 2
    assert not result['A'].isna().any()

    result2 = drop_nan(df, ['A', 'B'])
    assert len(result2) == 1
    assert not result2.isna().any().any()
