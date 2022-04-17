import torch

from tqdm import tqdm
from functools import cached_property

class Trainer:
    def __init__(self, model, learning_rate) -> None:
        self.model = model
        self.learning_rate = learning_rate

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.994
        )

    def train(self, train_x, train_y, epochs = 1000):
        pbar = tqdm(range(epochs), desc='Training Model Epochs')
        for epoch in pbar:
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
            self.scheduler.step()

            pbar.set_description(f"Training Model Epochs {round(loss.item(), 6)}")

            
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    def test(self, test_x, test_y):
        output = self.model(test_x)
        cost = self.criterion(output, test_y)
        print("mean squared error:", cost.item())

    def test_simulated(self, data_x, data_y, split_count, initial_value, look_forward):
        simulator = Simulator(self.model, data_x, data_y, split_count, look_forward, initial_value)
        return simulator.simulate()

    def to_file(self, path):
        torch.save(self.model.state_dict(), path)

    @classmethod
    def from_file(cls, model, path,):
        model.load_state_dict(torch.load(path))
        model.eval()
        return Trainer(model, 0)


class AccountSplit:
    def __init__(self, ) -> None:
        pass


class TradeData:
    def __init__(self, value, date, symbols) -> None:
        self.value = value
        self.date = date
        self.symbols = symbols


class SimulatorAccount:
    def __init__(self, initial) -> None:
        self.stored = initial
        self.shared = 0

    def add_shared(self, value):
        self.shared += value

    def set_shared(self, value):
        self.shared = value

    def set_stored(self, value):
        self.stored = value


class SimulatorAccounts:
    def __init__(self, initial, forward) -> None:
        self.forward = forward
        self.data = {x: SimulatorAccount(initial/forward) for x in range(forward)}

    def sell(self, day):
        for k in self.data:
            self.data[k].add_shared(self.data[day].stored/self.forward)
        self.data[day].set_stored(0)

    def shared(self, day):
        return self.data[day].shared

    def set_store(self, day, value):
        self.data[day].set_shared(0)
        return self.data[day].set_stored(value)

    def total_shared(self):
        total = 0
        for k in self.data:
            total += self.data[k].shared
        return total

    def total_value(self):
        total = 0
        for k in self.data:
            total += self.data[k].shared
            total += self.data[k].stored
        return total


class Simulator:
    def __init__(self, model, data_x, data_y, split, forward, initial) -> None:
        self.model = model
        self.data_x = data_x
        self.data_y = data_y
        self.split = split
        self.forward = forward
        self.accounts = SimulatorAccounts(initial, forward)
        self.history = []

    def simulate(self):
        for index in range(len(self.data_x)):
            self.simulate_day(index)
        return self.history

    def day_index(self, day):
        return day % self.forward

    def predict_day(self, day):
        return self.model(self.data_x[day])

    def top_pick_indices(self, day):
        return sorted(
            enumerate(self.predict_day(day)),
            key=lambda x: x[1],
            reverse=True,
        )[:self.split]

    def top_picks(self, day):
        picks = []
        for i, _ in self.top_pick_indices(day):
            picks.append(self.data_y[day].iloc[i])
        return picks

    def simulate_day(self, day):
        self.accounts.sell(self.day_index(day))

        new_value = 0
        symbols = []

        for pick in self.top_picks(day):
            if pick['gain'] < 3:
                new_value += pick['gain'] * self.accounts.shared(self.day_index(day))/self.split
            else:
                new_value += self.accounts.shared(self.day_index(day))/self.split
            symbols.append(pick["symbol"])
            date = pick["Date"]

        self.accounts.set_store(self.day_index(day), new_value)

        self.history.append(
            TradeData(
                value=self.accounts.total_value(),
                date=date,
                symbols=symbols
            )
        )