import argparse
import yfinance as yf
import numpy as np

# TODO
# add proper error handling for detected correctly formatted date inputs
# add option to select desired time frame

class MarketDataLoader:

    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '-T', '--tickers',
            nargs='+',
            default=[
                'AAPL',
                'MSFT',
                'GOOGL',
                'JPM',
                'XOM'
            ],
            help="Space-separated list of stock tickers."
        )

        parser.add_argument(
            '--interval', '-I',
            type=str,
            default='1d',
            choices=["1m", "5m", "15m", "60m", "1h", "1d"],
            help="Data sampling interval."
        )

        parser.add_argument(
            "--period", '-P',
            type=str, 
            default="2y",
            help="Lookback period (e.g., 60d, 1y, 2y). Ignored if --start is set."
        )

        parser.add_argument(
            "--start", 
            type=str, 
            default=None,
            help="Start date in YYYY-MM-DD format."
        )

        parser.add_argument(
            "--end", 
            type=str, 
            default=None,
            help="End date in YYYY-MM-DD format."
        )

        return parser.parse_args()

    def download(self):
        data = yf.download(
            tickers=self.args.tickers,
            start=self.args.start,
            end=self.args.end,
            period=self.args.period,
            interval=self.args.interval,
            auto_adjust=True,
            progress=True,
        )

        return data

    def market_caps(self):
        args = self.args

        market_caps = []

        for ticker in self.args.tickers:
            info = yf.Ticker(ticker).fast_info
            market_caps.append(info['market_cap'])

        return np.asarray(market_caps, dtype=float)
        
    def get_close_prices(self, data):
        return data['Close']