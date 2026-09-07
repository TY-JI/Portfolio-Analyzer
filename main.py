from yfinance import *

tickers = Tickers('MSFT AAPL')

print(tickers.tickers['MSFT'].balance_sheet)