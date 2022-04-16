from models import AccumulationDistribution

class Indicators:
    def __init__(self, session) -> None:
        self.session = session

    def cumulative(self):
        pass

    def calculate_ad(self, stock_price):
        ad_value = sum([self.calculate_single_ad(x) for x in self.cumulative(stock_price)])
        return AccumulationDistribution(
            stock_price=stock_price,
            value=ad_value,
        )

    def calculate_single_ad(self, x):
        return (((x.close - x.low) - (x.high - x.close)) / (x.high - x.low)) * x.volume
