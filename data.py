import argparse
import yfinance as yf

# TODO
# add proper error handling for detected correctly formatted date inputs
# add option to select desired time frame

class MarketDataLoader:

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
        args = self.parse_args()

        data = yf.download(
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            period=args.period,
            interval=args.interval,
            auto_adjust=True,
            progress=True,
        )

        return data
        
    def get_close_prices(self, data):
        return data['Close']