import pandas as pd
import numpy as np
import os
import tqdm

window = 175
train_set_path = './data/train_data'
valid_set_path = './data/valid_data'
if os.path.exists(train_set_path): os.remove(train_set_path)
if os.path.exists(valid_set_path): os.remove(valid_set_path)

def sampling(csv_path, save_path):
    files = os.listdir(csv_path)
    for file in tqdm.tqdm(files):
        df = pd.read_csv(csv_path + file)
        max_capacity = df.iloc[0, 0]
        if max_capacity < 0.4: continue
        voltage = df.iloc[:window, 1].to_numpy() 
        if window > df.shape[0]: voltage = np.pad(voltage, pad_width=(0, window - df.shape[0]), mode='edge')
        capacity = df.iloc[:window, 2].to_numpy() 
        if window > df.shape[0]: capacity = np.pad(capacity, pad_width=(0, window - df.shape[0]), mode='edge')
        sample = np.append(voltage, capacity)
        sample = np.append(sample, max_capacity)
        with open(save_path, 'a') as f:
            f.write(','.join(map(str, sample)))
            f.write('\n')

csv_path = './data/CALCE/csv_data/'
sampling(csv_path + 'CS2_35/', train_set_path)
sampling(csv_path + 'CS2_36/', train_set_path)
sampling(csv_path + 'CS2_37/', train_set_path)
sampling(csv_path + 'CS2_38/', valid_set_path)