from afin.ml.dataset import ZDataset, ZDatasetTransformer, DatasetSplit

import warnings
import pandas as pd
import plotly.express as px


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=pd.core.common.SettingWithCopyWarning)


from afin.ml.model import linearRegression, Ridge, NeuralNetwork
from afin.ml.trainer import Trainer

from tqdm import tqdm

def train():
    training_data = DatasetSplit.from_folder('/Users/adam/src/afin/training_data/training_data')
    test_data = DatasetSplit.from_folder('/Users/adam/src/afin/training_data/testing_data')

    for look_forward in range(1, 20):
        print(f'Testing look_forward={look_forward}')
        model = NeuralNetwork(91, 1)
        trainer = Trainer(model=model, learning_rate=0.01)
        trainer.train(training_data.data_x(), training_data.data_y(look_forward=look_forward), 1000)
        trainer.test(test_data.data_x(), test_data.data_y(look_forward=look_forward))
        trainer.to_file(f'/Users/adam/src/afin/models/output/{look_forward}_day.model')


def test():
    # dataset = ZDataset(
    #     '/Users/adam/src/afin/training_data/z_ticker_prices',
    # )
    # initial_value = 1000
    # split_lower = pd.Timestamp('now').floor('D') + pd.offsets.Day(-1 * 1 * 365)
    # training_data, test_data = dataset.split_dataset(split_lower)
    initial_value = 1000
    test_data = DatasetSplit.from_folder('/Users/adam/src/afin/training_data/testing_data')

    for look_forward in [3]:
        print(f'look_forward: {look_forward}')
        model = NeuralNetwork(91, 1)
        print('Loaded ')
        trainer = Trainer.from_file(model, f'/Users/adam/src/afin/models/output/{look_forward}_day.model')
        data_x, data_y = test_data.sim_data(look_forward=look_forward)
        split_data = []
        for i in range(3, 4):
            history = trainer.test_simulated(data_x, data_y, i, initial_value, look_forward)
            split_data.append((i, history))

        split, history = sorted(split_data, key=lambda x: x[1][-1].value)[-1]

        df = pd.DataFrame(dict(
            x = [z.date for z in history],
            y = [z.value for z in history],
            trades = [str(z.symbols) for z in history]
        ))
        fig = px.line(df, x="x", y="y",custom_data=['trades'], title=f'look_forward: {look_forward}, split: {split}, value: {history[-1].value}') 
        fig.update_traces(
            hovertemplate="<br>".join([
                "ColX: %{x}",
                "ColY: %{y}",
                "Col1: %{customdata[0]}",
            ])
        )
        fig.show()
    




if __name__ == '__main__':
    test()