

import yfinance as yf
from datetime import datetime

import pandas as pd
from pandas_datareader.data import DataReader


def getLastYearsStocks(stock_tag):
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    return yf.download(stock_tag, start, end)

STOCKS_LIST = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']

def getStocksDF(stocks_list=STOCKS_LIST):
    stocks = [getLastYearsStocks(stock) for stock in STOCKS_LIST]
    # print(stocks)
    stocks_dict = {}
    for tag, stock in zip(STOCKS_LIST, stocks):
        stocks_dict[tag] = stock

    df = pd.concat(stocks_dict, axis=0)
    return df