from afin.ml.dataset import ZDataset, ZDatasetTransformer

import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from afin.ml.model import linearRegression
from afin.ml.trainer import Trainer


from datetime import date

def main():
    dataset = ZDataset(
        '/Users/adam/src/afin/training_data/z_ticker_prices',
    )
    split_lower = pd.Timestamp('now').floor('D') + pd.offsets.Day(-1 * 1 * 365)
    training_data, test_data = dataset.split_dataset(split_lower)

    for look_forward in range(1, 2):
        print(f'Testing look_forward={look_forward}')
        model = linearRegression(91, 1)
        trainer = Trainer(model=model, learning_rate=0.0001)
        trainer.train(training_data.data_x(), training_data.data_y(look_forward=look_forward))
        trainer.test(test_data.data_x(), test_data.data_y(look_forward=look_forward))
        trainer.to_file(f'/Users/adam/src/afin/models/output/{look_forward}_day.model')


if __name__ == '__main__':
    main()