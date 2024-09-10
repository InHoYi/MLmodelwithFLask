import pandas as pd
import numpy as np

def extract_data(data, include_defrost = None):
    if include_defrost:
        return data[['attribute_1_value', 'attribute_2_value']]
    else:
        return data[['attribute_1_value']]

def get_min_temp(aFeature):
    result = aFeature.min()
    return result

def get_max_temp(aFeature):
    result = aFeature.max()
    return result

def scale_data(aFeature):
    scaled_aFeature = aFeature.copy()
    min_val = scaled_aFeature.min()
    max_val = scaled_aFeature.max()
    scaled_aFeature = (scaled_aFeature - min_val) / (max_val - min_val)
    return scaled_aFeature

def load_data( path , scale = None, include_deforst = None):
    data = pd.read_csv( path )
    # data['measure_dtm'] = pd.to_datetime(data['measure_dtm'])
    # data.replace({"OFF": 0, "ON": 1}, inplace=True)
    # data = extract_data(data, include_deforst)
    if scale:
        return scale_data(data)
    else:
        return data

def train_test_split(data, train_ratio = None, valid_ratio = None): # 0.6 / 0.75
    total_length = data.shape[0]
    train_test_split = int(total_length * train_ratio)
    train_valid_split = int(train_test_split * valid_ratio)
    train = data[:train_valid_split]
    valid = data[train_valid_split:train_test_split]
    test = data[train_test_split:]
    return train, valid, test

def create_dataset(data, shuffle=None, window_size=None, number_of_features=None):
    # data = data.values
    X, y = [], []
    for i in range(len(data) - window_size):
        end_ix = i + window_size
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    if shuffle:
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
    return X, y