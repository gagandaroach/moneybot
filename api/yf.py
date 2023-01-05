

import yfinance as yf
from datetime import datetime

import pandas as pd


def getLastYearsStocks(stock_tag):
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    return yf.download(stock_tag, start, end)

stock_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
stocks = [getLastYearsStocks(stock) for stock in stock_list]
# print(stocks)
stocks_dict = {}
for tag, stock in zip(stock_list, stocks):
    stocks_dict[tag] = stock

df = pd.concat(stocks_dict, axis=0)
print(df.tail(10))
