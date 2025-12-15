import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder

def one_hot(dataset, cols):
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(dataset[cols])
    return encoder

def one_hot_transform(dataset, cols, encoder):
    encoded = encoder.transform(dataset[cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols))
    
    dataset = dataset.drop(cols, axis=1).reset_index(drop=True)
    return pd.concat([dataset, encoded_df], axis=1)

def label(dataset, cols):
    encoder = LabelEncoder()
    dataset[cols] = encoder.fit_transform(dataset[cols])
    return dataset

def target(dataset, cols):
    encoder = TargetEncoder()
    dataset[cols] = encoder.fit_transform(dataset[cols])
    return dataset
