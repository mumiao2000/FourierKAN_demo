import pandas as pd
import numpy as np
import os

path = './data/CALCE/csv_data/CS2_35/'
save_path = './plot/'
file_name_list = os.listdir(path)

length = [pd.read_csv(path + file_name, encoding='utf-8').shape[0] for file_name in file_name_list]
print(len(length))
print('min  length: ', min(length))
print('max  length: ', max(length))
print('mean length: ', np.mean(length))


max_capacity = [pd.read_csv(path + file_name, encoding='utf-8').iloc[0, 0] for file_name in file_name_list]
print('min  max_capacity: ', min(max_capacity))
print('max  max_capacity: ', max(max_capacity))
print('mean max_capacity: ', np.mean(max_capacity))
print('max_capacity interval: ', max(max_capacity) - min(max_capacity))