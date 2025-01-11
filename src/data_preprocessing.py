import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['NormAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Amount', 'Time'], axis=1)
    return data

def split_data(data):
    fraud_data = data[data['Class'] == 1]
    non_fraud_data = data[data['Class'] == 0]
    return fraud_data, non_fraud_data