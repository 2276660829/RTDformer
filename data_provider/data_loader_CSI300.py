import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class StockDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',num_stock=155,print_debug=False):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.num_stock=num_stock
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.print_debug = print_debug 

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
    def __read_data__(self):
        
        self.scaler = StandardScaler()
        df_raw = torch.load(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)


        df_raw =df_raw[cols + ['close']]
        
        unique_dates = sorted(df_raw.index.get_level_values('date').unique())
        
        # DATE
        train_end_date = '2017-08-30'
        vali_end_date = '2021-06-16'
        
        num_train = unique_dates.index((train_end_date)) + 1
        
        
        num_vali = unique_dates.index((vali_end_date)) + 1
        num_test = len(unique_dates) - num_train - num_vali

        border1s = [0, num_train - self.seq_len, num_vali- self.seq_len]

        border2s = [num_train,  num_vali, int(len(df_raw)/self.num_stock)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        


        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[0:]
            df_data = df_raw[cols_data]  

        data = df_data.values


        

        
        df_stamp = unique_dates[border1:border2]
        df_stamp = pd.to_datetime(df_stamp)
        
        data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        
        # 获取指定范围的日期
        selected_dates = unique_dates[border1:border2]
        
#         print(selected_dates[-1]) #'2024-04-30')

#         filtered_df = df_raw.loc[(slice(None), selected_dates, slice(None)), :]
        
        filtered_df = df_raw.loc[(selected_dates, slice(None)), :]

        self.data_x = filtered_df
        self.data_y = filtered_df

        
        self.data_stamp = data_stamp
        

    def __getitem__(self, index):
#         print(index)
        adjusted_index = index *1
        
        s_begin = adjusted_index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len

        r_end = r_begin + self.label_len + self.pred_len
        unique_dates_1=sorted(self.data_x.index.get_level_values('date').unique())
        
        selected_dates_x = unique_dates_1[s_begin:s_end]
        filtered_df_x = self.data_x.loc[(selected_dates_x, slice(None)), :]
        
        
        unique_codes = sorted(filtered_df_x.index.get_level_values('code').unique())
        filtered_df_x_code = filtered_df_x.sort_index(level='code')

        
              
        seq_x=filtered_df_x_code.values.reshape(self.num_stock, self.seq_len, filtered_df_x_code.shape[1])

        
        selected_dates_y=unique_dates_1[r_begin:r_end]

        

    
        filtered_df_y=self.data_y.loc[( selected_dates_y, slice(None)), :]
        
        unique_codes = sorted(filtered_df_y.index.get_level_values('code').unique())
        filtered_df_y_code = filtered_df_y.sort_index(level='code')
        
        seq_y=filtered_df_y_code.values.reshape(self.num_stock, int(len(filtered_df_y_code)/self.num_stock), filtered_df_y_code.shape[1])
        
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = np.tile(seq_x_mark, (self.num_stock, 1, 1))
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_y_mark = np.tile(seq_y_mark, (self.num_stock, 1, 1))

        seq_x = np.array(seq_x, dtype=np.float32)
        seq_x = torch.tensor(seq_x)
        seq_y = np.array(seq_y, dtype=np.float32)
        seq_y = torch.tensor(seq_y)
        seq_x_mark = np.array(seq_x_mark, dtype=np.float32)
        seq_x_mark = torch.tensor(seq_x_mark)
        seq_y_mark = np.array(seq_y_mark, dtype=np.float32)
        seq_y_mark = torch.tensor(seq_y_mark)
        
        

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        



        return (len(self.data_x)//self.num_stock - self.seq_len - self.pred_len + 1)//1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
