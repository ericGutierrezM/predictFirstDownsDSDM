from sklearn.preprocessing import StandardScaler

def scaler(dataset, num_cols):
    scaler = StandardScaler()
    dataset[num_cols] = scaler.fit_transform(dataset[num_cols])
    return dataset, scaler

def scaler_transform(dataset, num_cols, scaler):
    dataset[num_cols] = scaler.transform(dataset[num_cols])
    return dataset