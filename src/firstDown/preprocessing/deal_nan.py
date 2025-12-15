import pandas as pd

def search_nan(dataset): # return a table with the columns with nan, number of nans

    var_list = []
    rows_nan_list = []
    rows_other_list = []
    rows_total_list = []

    for feature in dataset.columns:
        var_list.append(feature)
        rows_nan_list.append(dataset[feature].isna().sum())
        rows_other_list.append(len(dataset[feature]) - dataset[feature].isna().sum())
        rows_total_list.append(len(dataset[feature]))
    
    nan_table = pd.DataFrame({
        'var': var_list,
        'rows_nan': rows_nan_list,
        'rows_other': rows_other_list,
        'rows_total': rows_total_list    
    })

    return nan_table


def replace_nan(dataset, cols, method, num=0, txt='missing'): # with an argument on how to deal with them

    match method:
        case 'mean':
            dataset[cols] = dataset[cols].fillna(dataset[cols].mean())
        case 'median':
            dataset[cols] = dataset[cols].fillna(dataset[cols].median())
        case 'min':
            dataset[cols] = dataset[cols].fillna(dataset[cols].min())
        case 'max':
            dataset[cols] = dataset[cols].fillna(dataset[cols].max())
        case 'num':
            dataset[cols] = dataset[cols].fillna(num)
        case 'text':
            dataset[cols] = dataset[cols].fillna(txt)
        case _:
            print('Error in specifications. No rows were replaced.')
    
    return dataset

def drop_nan(dataset, cols):

    return dataset.dropna(subset=cols)