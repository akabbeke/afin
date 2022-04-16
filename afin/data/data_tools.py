import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



import yfinance as yf
import datetime
import time
import requests

from tqdm.contrib.concurrent import process_map
import io

from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from ta import add_all_ta_features
from ta.utils import dropna

from .models import Stock, StockPrice
from .db import create_db, engine

from multiprocessing import Pool

def fetch_companies():
    url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
    s = requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode('utf-8')))

def insert_tickers():
    companies = fetch_companies()
    tickers = []
    for _, row in companies.iterrows():
        tickers.append(Stock(
            symbol=row['Symbol'],
            name=row['Company Name'],
            security_name=row['Security Name'],
            category=row['Market Category'],
            test_issue=row['Test Issue']!='N',
            status=row['Financial Status']!='N',
            round_lot_size=int(row['Round Lot Size']),
        ))
    
    with Session(engine) as session:
        session.add_all(tickers)
        session.commit()

def fetch_prices(symbol):

    start_date = datetime.datetime(2010,1,1)
    end_date = datetime.datetime.today()

    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        show_errors=False,
    )

    data = dropna(data)

    data = add_all_ta_features(
        data,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )

    data['trend_cci'] = data['trend_cci'].fillna(0)
    data['trend_visual_ichimoku_a'] = data['trend_visual_ichimoku_a'].fillna(0)
    data['trend_psar_up'] = data['trend_psar_up'].fillna(0)
    data['trend_psar_down'] = data['trend_psar_down'].fillna(0)

    data.to_csv(f'/Users/adam/src/afin/training_data/ticker_prices/{symbol.upper()}.csv')


def get_stock_data(symbol):
    try:
        return fetch_prices(symbol)
    except:
        pass


def insert_prices():
    with Session(engine) as session:
        results = session.query(Stock).all()

    symbols = [x.symbol for x in results]
    process_map(get_stock_data, symbols, chunksize=1)

def main():
    insert_prices()


if __name__ == '__main__':
    insert_prices()