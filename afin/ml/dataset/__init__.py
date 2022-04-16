import imp
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


from os import listdir
from os.path import isfile, join

from datetime import timedelta

from . import utils


class DatframeLoader:
    def __init__(self, path) -> None:
        self.path = path
        self.dataframe = pd.read_csv(self.path)
        self.dataframe = self.dataframe.drop(columns=['volume_adi'])


class ZDatasetTransformer:
    def __init__(self, in_path, out_path) -> None:
        self.in_path = in_path
        self.out_path = out_path

    @property
    def non_z_columns(self):
        return ['Date']

    def z_scored(self):
        only_files = sorted([f for f in listdir(self.in_path) if isfile(join(self.in_path, f)) and '.csv' in f])

        input_data = []
        print('Splitting by symbol')
        for file_name in tqdm(only_files):
            input_data.append([
                f'{self.in_path}/{file_name}',
                f'{self.out_path}/{file_name}',
                self.non_z_columns, 
                file_name.split('.')[0]
            ])
        print(f'Split into {len(input_data)} items')
        process_map(utils.intermediate, input_data, max_workers=10, chunksize=10, desc='Creating Z-Score Files')


class ZDataset:
    def __init__(self, in_path) -> None:
        self.in_path = in_path
        self.dataframe = self.load_dataframe()
        self.dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.dataframe = self.dataframe.dropna()

    def load_dataframe(self):
        only_files = sorted([f'{self.in_path}/{f}' for f in listdir(self.in_path) if isfile(join(self.in_path, f)) and '.csv' in f])
        return pd.concat(process_map(utils.load_dataframe, only_files, chunksize=10, desc='Reading Z-Score Files'))

    def split_dataset(self, split_lower):
        split_upper = timedelta(days=50) + split_lower
        lower = DatasetSplit(self.dataframe[(self.dataframe['Date'] < split_lower)])
        upper = DatasetSplit(self.dataframe[(self.dataframe['Date'] > split_upper)])
        return lower, upper

    def get_feature_columns(self):
        excluded = ['symbol', 'change_open', 'Date', 'index']
        return sorted([x for x in self.dataframe.columns if x not in excluded and 'future_' not in x])

    def df_to_tensor(self, dataframe, look_forward):
        col_names = self.get_feature_columns()

        data_x = Variable(torch.from_numpy(dataframe[col_names].to_numpy(dtype=np.float64))).float()
        data_y = Variable(torch.from_numpy(dataframe[f'future_{look_forward}_day'].to_numpy(dtype=np.float64))).float()
        return data_x, data_y.view((len(data_y), 1))

    def df_to_test_tensor(self, look_forward):
        col_names = self.get_feature_columns(self.dataframe)
        X = Variable(torch.from_numpy(self.dataframe[col_names].to_numpy())).float()
        y = self.dataframe[[f'future_{look_forward}_day', 'symbol', 'Date']]

        return X, y


class DatasetSplit:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    def get_feature_columns(self):
        excluded = ['symbol', 'change_open', 'Date', 'index']
        return sorted([x for x in self.dataframe.columns if x not in excluded and 'future_' not in x])
    
    def dataframe_x(self):
        return self.dataframe[self.get_feature_columns()]

    def dataframe_y(self, look_forward):
        return self.dataframe[f'future_{look_forward}_day']

    def data_x(self):
        dataframe = self.dataframe_x()
        if(dataframe.isnull().values.any()):
            raise Exception()
        return Variable(torch.from_numpy(dataframe.to_numpy(dtype=np.float64))).float()

    def data_y(self, look_forward):
        dataframe = self.dataframe_y(look_forward)
        if(dataframe.isnull().values.any()):
            raise Exception()
        data_y = Variable(torch.from_numpy(dataframe.to_numpy(dtype=np.float64))).float()
        return data_y.view((len(data_y), 1))
