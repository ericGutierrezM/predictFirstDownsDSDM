from sklearn.preprocessing import StandardScaler

def scale(dataset, num_cols): # scale numeric variables
    scaler = StandardScaler()
    dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

    return dataset