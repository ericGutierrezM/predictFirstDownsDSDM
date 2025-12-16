import pandas as pd
from src.firstDown.preprocessing.scale import scaler

def test_scaler_numeric_columns_scaled():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y'] 
    })

    num_cols = ['A', 'B']
    result, _ = scaler(df.copy(), num_cols)

    for col in num_cols:
        assert abs(result[col].mean()) < 1e-10
        assert abs(result[col].std(ddof=0) - 1) < 1e-10

    assert (result['C'] == df['C']).all()


def test_scaler_single_column():
    df = pd.DataFrame({'A': [1, 2, 3, 4]})
    result, _ = scaler(df.copy(), ['A'])

    assert abs(result['A'].mean()) < 1e-10
    assert abs(result['A'].std(ddof=0) - 1) < 1e-10
