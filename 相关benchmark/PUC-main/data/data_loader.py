import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def data_split(root_path, data_name, is_valid=False):
    if data_name == 'criteo':
        df = pd.read_csv(os.path.join(root_path, 'data', 'criteo-uplift-v2.1.csv'))
        train, test  = train_test_split(df, test_size=0.2, random_state=0, stratify=df['treatment'])
        train_data = train.drop(columns = ['conversion', 'exposure'])
        test_data = test.drop(columns = ['conversion', 'exposure'])
        valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=0, stratify=test_data['treatment'])

        np.savez(os.path.join(root_path, 'data', 'criteo'), train=np.array(train_data), valid=np.array(valid_data), test=np.array(test_data))

        

def load_data(root_path, data_name, is_valid=False):

    data_file = os.path.join(root_path, 'data', data_name+'.npz')

    if is_valid:
        data = np.load(data_file)
        train_data = data['train']
        valid_data = data['valid']
        test_data = data['test']
        return train_data, valid_data, test_data
    else:
        data = np.load(data_file)
        train_data = data['train']
        test_data = data['test']
        return train_data, test_data
