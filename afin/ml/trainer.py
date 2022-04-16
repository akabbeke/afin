import torch

from tqdm import tqdm
from functools import cached_property

class Trainer:
    def __init__(self, model, learning_rate) -> None:
        self.model = model
        self.learning_rate = learning_rate

    @cached_property
    def criterion(self):
        return torch.nn.MSELoss()

    @cached_property
    def optimizer(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def train(self, train_x, train_y, epochs = 1000):
        print(list(self.model.parameters()))
        for epoch in tqdm(range(epochs), desc='Training Model Epochs'):
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(train_x)
            # get loss for the predicted output
            loss = self.criterion(outputs, train_y)

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            self.optimizer.step()

            
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    def test(self, test_x, test_y):
        output = self.model(test_x)
        cost = self.criterion(output, test_y)
        print("mean squared error:", cost.item())

    def test_simulated(model, look_forward, start_value, dataframe):
        date_slices = load_test_dataset('2021-06-01', '2021-08-01', dataframe, look_forward)
        total_value = start_value
        split_count = 10
        out_data  = []
        for test_x, test_y, date in date_slices:
            
            new_value = 0
            output = model(test_x)
            sourted_output = sorted(enumerate(output), key=lambda x: x[1], reverse=True)
            for i, increase in sourted_output[:split_count]:
                row_data = test_y.iloc[i]
                increase = row_data[f'future_{look_forward}_day']
                # print(f'  {row_data["Date"]} - {row_data["symbol"]} - increase {increase} - buy {total_value/5} - sell {increase * total_value/5} - earn {(increase * total_value/5) - total_value/5}')
                new_value += increase * total_value/split_count
            print(date, new_value/total_value)
            total_value = new_value
            out_data.append([date, total_value])
        return out_data


    def to_file(self, path):
        torch.save(self.model.state_dict(), path)

    @classmethod
    def from_file(cls, model, path):
        model.load_state_dict(torch.load(path))
        model.eval()
        return Trainer(model)