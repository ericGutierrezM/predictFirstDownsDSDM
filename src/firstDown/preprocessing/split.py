from sklearn.model_selection import train_test_split

def split(dataset, y_col, test_size): # train test split
    return train_test_split(dataset.drop(y_col, axis=1), dataset[y_col], test_size=test_size, random_state=11)