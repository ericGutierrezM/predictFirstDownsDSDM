import pandas as pd

def drop_penalties(dataset, penalty_col):    # drop first downs due to penalties
    dataset = dataset[dataset[penalty_col]==0.0]
    return dataset

def drop_control_rows(dataset, control_col, filter_out=['GAME','END QUARTER 1', 'END QUARTER 2', 'END QUARTER 3', 'END QUARTER 4', 'END GAME']): # drop rows for game initilization
    dataset = dataset[dataset[control_col].isin(filter_out)==False]
    return dataset