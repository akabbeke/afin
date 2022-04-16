import pandas as pd
import numpy as np


def continuous_z_col(column):
    output = []
    for i, value in enumerate(column):
        lower = column[0:i].std(ddof=0)
        upper = (value - column[0: i].mean())
        if lower == 0:
            output.append(np.nan)
        else:
            output.append(upper/lower)
    return output


def continuous_z(input_data):
    from . import DatframeLoader
    in_path, out_path, non_z_columns, symbol = input_data
    dataframe = DatframeLoader(in_path).dataframe
    dataframe = dataframe.dropna()
    data = {}

    for col_name in [x for x in dataframe.columns if x not in non_z_columns]:
        data[col_name] = continuous_z_col(dataframe[col_name])
    
    z_dataframe = pd.DataFrame(data)

    for col_name in non_z_columns:
        z_dataframe[col_name] = dataframe[col_name]

    z_dataframe['symbol'] = symbol

    z_dataframe = z_dataframe.dropna()
    z_dataframe.to_csv(out_path)

def compute_continuous_z(input_data):
    from . import DatframeLoader
    in_path, out_path, non_z_columns, symbol = input_data
    dataframe = DatframeLoader(in_path).dataframe
    dataframe = dataframe.dropna()

    z_df = pd.DataFrame()
    
    for col in [x for x in dataframe.columns if x not in non_z_columns]:
        z_df[col] = (dataframe[col] - dataframe[col].expanding().mean())/dataframe[col].expanding().std(ddof=0)
    
    for offset in range(0, 31):
        z_df[f'future_{offset}_day'] =(dataframe['Close'].shift(-1*offset)/dataframe['Close'])

    z_df['Date'] = dataframe['Date']
    z_df['symbol'] = symbol
    z_df = z_df.dropna()
    z_df = z_df.copy()
    z_df.to_csv(out_path)


def intermediate(input_data):
    compute_continuous_z(input_data)


def load_dataframe(in_path):
    dataframe = pd.read_csv(in_path, index_col=0)
    dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%Y-%m-%d')
    return dataframe