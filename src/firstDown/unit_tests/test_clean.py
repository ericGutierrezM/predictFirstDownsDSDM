import pandas as pd

from src.firstDown.preprocessing.clean import (
    drop_penalties,
    drop_control_rows
)

def test_drop_penalties_removes_rows():
    df = pd.DataFrame({
        'penalty': [0.0, 1.0, 0.0]
    })

    result = drop_penalties(df, 'penalty')

    assert len(result) == 2
    assert (result['penalty'] == 0.0).all()

def test_drop_penalties_no_change_when_no_penalties():
    df = pd.DataFrame({
        'penalty': [0.0, 0.0]
    })

    result = drop_penalties(df, 'penalty')

    assert len(result) == len(df)

def test_drop_control_rows_removes_default_controls():
    df = pd.DataFrame({
        'play_desc': [
            'RUN',
            'END QUARTER 1',
            'PASS',
            'END GAME'
        ]
    })

    result = drop_control_rows(df, 'play_desc')

    assert 'END QUARTER 1' not in result['play_desc'].values
    assert 'END GAME' not in result['play_desc'].values
    assert len(result) == 2

def test_drop_control_rows_custom_filter():
    df = pd.DataFrame({
        'control': ['A', 'B', 'C']
    })

    result = drop_control_rows(df, 'control', filter_out=['B'])

    assert 'B' not in result['control'].values
    assert len(result) == 2
